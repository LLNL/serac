// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file quadrilateral_H1.inl
 *
 * @brief Specialization of finite_element for H1 on quadrilateral geometry
 */

// this specialization defines shape functions (and their gradients) that
// interpolate at Gauss-Lobatto nodes for the appropriate polynomial order
//
// note: mfem assumes the parent element domain is [0,1]x[0,1]
// for additional information on the finite_element concept requirements, see finite_element.hpp
/// @cond
template <int p, int c>
struct finite_element<Geometry::Quadrilateral, H1<p, c> > {
  static constexpr auto geometry   = Geometry::Quadrilateral;
  static constexpr auto family     = Family::H1;
  static constexpr int  components = c;
  static constexpr int  dim        = 2;
  static constexpr int  n          = (p + 1);
  static constexpr int  ndof       = (p + 1) * (p + 1);

  using residual_type =
      typename std::conditional<components == 1, tensor<double, ndof>, tensor<double, ndof, components> >::type;

  using dof_type = tensor<double, c, p + 1, p + 1>;

  /**
   * @brief this type is used when calling the batched interpolate/integrate
   *        routines, to provide memory for calculating intermediates
   */
  template <int q>
  using cache_type = tensor<double, 2, n, q>;

  template <int q>
  using cpu_batched_values_type = tensor<tensor<double, c>, q, q>;

  template <int q>
  using cpu_batched_derivatives_type = tensor<tensor<double, c, 2>, q, q>;

  /*

    interpolation nodes and their associated numbering:

        linear
    2-----------3
    |           |
    |           |
    |           |
    |           |
    |           |
    0-----------1


      quadratic
    6-----7-----8
    |           |
    |           |
    3     4     5
    |           |
    |           |
    0-----1-----2


        cubic
    12-13--14--15
    |           |
    8   9  10  11
    |           |
    4   5   6   7
    |           |
    0---1---2---3

  */

  SERAC_HOST_DEVICE static constexpr tensor<double, ndof> shape_functions(tensor<double, dim> xi)
  {
    auto N_xi  = GaussLobattoInterpolation<p + 1>(xi[0]);
    auto N_eta = GaussLobattoInterpolation<p + 1>(xi[1]);

    int count = 0;

    tensor<double, ndof> N{};
    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p + 1; i++) {
        N[count++] = N_xi[i] * N_eta[j];
      }
    }
    return N;
  }

  SERAC_HOST_DEVICE static constexpr tensor<double, ndof, dim> shape_function_gradients(tensor<double, dim> xi)
  {
    auto N_xi   = GaussLobattoInterpolation<p + 1>(xi[0]);
    auto N_eta  = GaussLobattoInterpolation<p + 1>(xi[1]);
    auto dN_xi  = GaussLobattoInterpolationDerivative<p + 1>(xi[0]);
    auto dN_eta = GaussLobattoInterpolationDerivative<p + 1>(xi[1]);

    int count = 0;

    tensor<double, ndof, dim> dN{};
    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p + 1; i++) {
        dN[count++] = {dN_xi[i] * N_eta[j], N_xi[i] * dN_eta[j]};
      }
    }
    return dN;
  }

  template <int q>
  static auto interpolate(const dof_type& X, const tensor<double, dim, dim, q, q>& jacobians,
                          const TensorProductQuadratureRule<q>&)
  {
    // we want to compute the following:
    //
    // X_q(u, v) := (B(u, i) * B(v, j) ) * X_e(i, j)
    //
    // where
    //   X_q(u, v) are the quadrature-point values at position {u, v},
    //   B(u, i) is the i^{th} 1D interpolation/differentiation (shape) function,
    //           evaluated at the u^{th} 1D quadrature point, and
    //   X_e(i, j) are the values at node {i, j} to be interpolated
    //
    // this algorithm carries out the above calculation in 2 steps:
    //
    // A(dy, qx)  := B(qx, dx) * X_e(dy, dx)
    // X_q(qy, qx) := B(qy, dy) * A(dy, qx)

    static constexpr auto points1D = GaussLegendreNodes<q>();
    static constexpr auto B        = [=]() {
      tensor<double, q, n> B_{};
      for (int i = 0; i < q; i++) {
        B_[i] = GaussLobattoInterpolation<n>(points1D[i]);
      }
      return B_;
    }();

    static constexpr auto G = [=]() {
      tensor<double, q, n> G_{};
      for (int i = 0; i < q; i++) {
        G_[i] = GaussLobattoInterpolationDerivative<n>(points1D[i]);
      }
      return G_;
    }();

    cache_type<q> A;

    serac::tuple<cpu_batched_values_type<q>, cpu_batched_derivatives_type<q> > values_and_derivatives{};

    for (int i = 0; i < c; i++) {
      for (int dy = 0; dy < n; dy++) {
        for (int qx = 0; qx < q; qx++) {
          double sum[2]{};
          for (int dx = 0; dx < n; dx++) {
            sum[0] += B(qx, dx) * X(i, dy, dx);
            sum[1] += G(qx, dx) * X(i, dy, dx);
          }
          A(0, dy, qx) = sum[0];
          A(1, dy, qx) = sum[1];
        }
      }

      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          double sum[3]{};
          for (int dy = 0; dy < n; dy++) {
            sum[0] += B(qy, dy) * A(0, dy, qx);
            sum[1] += B(qy, dy) * A(1, dy, qx);
            sum[2] += G(qy, dy) * A(0, dy, qx);
          }
          serac::get<0>(values_and_derivatives)(qy, qx)(i)    = sum[0];
          serac::get<1>(values_and_derivatives)(qy, qx)(i, 0) = sum[1];
          serac::get<1>(values_and_derivatives)(qy, qx)(i, 1) = sum[2];
        }
      }
    }
    for (int qy = 0; qy < q; qy++) {
      for (int qx = 0; qx < q; qx++) {
        tensor< double, dim, dim > J;
        for (int row = 0; row < dim; row++) {
          for (int col = 0; col < dim; col++) {
            J[row][col] = jacobians(col, row, qy, qx);
          }
        }
        auto grad_u = serac::get<1>(values_and_derivatives)(qy, qx);

        serac::get<1>(values_and_derivatives)(qy, qx) = dot(grad_u, inv(J));
      }
    }

    return values_and_derivatives;
  }

  template <int q>
  static void integrate(cpu_batched_values_type<q>& sources, cpu_batched_derivatives_type<q>& fluxes,
                        const tensor<double, dim, dim, q, q>& jacobians, const TensorProductQuadratureRule<q>&,
                        dof_type&                             element_residual)
  {
    static constexpr auto points1D  = GaussLegendreNodes<q>();
    static constexpr auto weights1D = GaussLegendreWeights<q>();
    static constexpr auto B         = [=]() {
      tensor<double, q, n> B_{};
      for (int i = 0; i < q; i++) {
        B_[i] = GaussLobattoInterpolation<n>(points1D[i]);
      }
      return B_;
    }();

    static constexpr auto G = [=]() {
      tensor<double, q, n> G_{};
      for (int i = 0; i < q; i++) {
        G_[i] = GaussLobattoInterpolationDerivative<n>(points1D[i]);
      }
      return G_;
    }();

    cache_type<q> A{};

    for (int qy = 0; qy < q; qy++) {
      for (int qx = 0; qx < q; qx++) {
        tensor< double, dim, dim > J_T;
        for (int row = 0; row < dim; row++) {
          for (int col = 0; col < dim; col++) {
            J_T[row][col] = jacobians(row, col, qy, qx);
          }
        }
        auto dv         = det(J_T) * weights1D[qx] * weights1D[qy];
        sources(qy, qx) = sources(qy, qx) * dv;
        fluxes(qy, qx)  = dot(fluxes(qy, qx), inv(J_T)) * dv;
      }
    }

    for (int i = 0; i < c; i++) {
      for (int dx = 0; dx < n; dx++) {
        for (int qy = 0; qy < q; qy++) {
          double sum[2]{};
          for (int qx = 0; qx < q; qx++) {
            sum[0] += B(qx, dx) * sources(qy, qx)[i];
            sum[0] += G(qx, dx) * fluxes(qy, qx)[i][0];
            sum[1] += B(qx, dx) * fluxes(qy, qx)[i][1];
          }
          A(0, dx, qy) = sum[0];
          A(1, dx, qy) = sum[1];
        }
      }

      for (int dx = 0; dx < n; dx++) {
        for (int dy = 0; dy < n; dy++) {
          double sum = 0.0;
          for (int qy = 0; qy < q; qy++) {
            sum += B(qy, dy) * A(0, dx, qy);
            sum += G(qy, dy) * A(1, dx, qy);
          }
          element_residual(i, dy, dx) += sum;
        }
      }
    }
  }
};
/// @endcond

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

    tensor< double, c, q, q> value{};
    tensor< double, c, dim, q, q> gradient{};

    for (int i = 0; i < c; i++) {
      A[0] = contract<1, 1>(X[i], B);
      A[1] = contract<1, 1>(X[i], G);

      value(i)      = contract<0, 1>(A[0], B);
      gradient(i,0) = contract<0, 1>(A[1], B);
      gradient(i,1) = contract<0, 1>(A[0], G);
    }

    tensor< tensor< double, c >, q, q > value_T{};
    tensor< tensor< double, c, dim >, q, q > gradient_T{};

    for (int qy = 0; qy < q; qy++) {
      for (int qx = 0; qx < q; qx++) {
        tensor<double, dim, dim> J;
        for (int row = 0; row < dim; row++) {
          for (int col = 0; col < dim; col++) {
            J[row][col] = jacobians(col, row, qy, qx);
          }
        }

        for (int i = 0; i < c; i++) {
          value_T(qy, qx)[i] = value(i, qy, qx);
          for (int j = 0; j < dim; j++) {
            gradient_T(qy, qx)[i][j] = gradient(i, j, qy, qx);
          }
        }

        gradient_T(qy, qx) = dot(gradient_T(qy, qx), inv(J));
      }
    }
 
    return tuple{value_T, gradient_T};
  }

  template <int q>
  static void integrate(cpu_batched_values_type<q>& source_T, cpu_batched_derivatives_type<q>& flux_T,
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

    tensor< double, c, q, q> source{};
    tensor< double, c, dim, q, q> flux{};

    for (int qy = 0; qy < q; qy++) {
      for (int qx = 0; qx < q; qx++) {
        tensor<double, dim, dim> J_T;
        for (int row = 0; row < dim; row++) {
          for (int col = 0; col < dim; col++) {
            J_T[row][col] = jacobians(row, col, qy, qx);
          }
        }
        auto dv = det(J_T) * weights1D[qx] * weights1D[qy];

        source_T(qy, qx) = source_T(qy, qx) * dv;
        flux_T(qy, qx)  = dot(flux_T(qy, qx), inv(J_T)) * dv;

        for (int i = 0; i < c; i++) {
          source(i, qy, qx) = source_T(qy, qx)[i];
          for (int j = 0; j < dim; j++) {
            flux(i, j, qy, qx) = flux_T(qy, qx)[i][j];
          }
        }
      }
    }

    cache_type<q> A{};

    for (int i = 0; i < c; i++) {
      A[0] = contract< 1, 0 >(source[i], B) + contract< 1, 0 >(flux(i, 0), G);
      A[1] = contract< 1, 0 >(flux(i, 1), B);

      element_residual(i) += contract< 0, 0 >(A[0], B) + contract< 0, 0 >(A[1], G);
    }

  }

#if defined(__CUDACC__)

  template <int q>
  static SERAC_DEVICE auto interpolate(const dof_type& X, const tensor<double, dim, dim>& J,
                                       const TensorProductQuadratureRule<q>& rule, cache_type<q>& A)
  {
    int tidx = threadIdx.x % q;
    int tidy = threadIdx.x / q;

    static constexpr auto points1D = GaussLegendreNodes<q>();
    static constexpr auto B_       = [=]() {
      tensor<double, q, n> B{};
      for (int i = 0; i < q; i++) {
        B[i] = GaussLobattoInterpolation<n>(points1D[i]);
      }
      return B;
    }();

    static constexpr auto G_ = [=]() {
      tensor<double, q, n> G{};
      for (int i = 0; i < q; i++) {
        G[i] = GaussLobattoInterpolationDerivative<n>(points1D[i]);
      }
      return G;
    }();

    __shared__ tensor<double, q, n> B;
    __shared__ tensor<double, q, n> G;
    for (int entry = threadIdx.x; entry < n * q; entry += q * q) {
      int i   = entry % n;
      int j   = entry / n;
      B(j, i) = B_(j, i);
      G(j, i) = G_(j, i);
    }
    __syncthreads();

    tuple<tensor<double, c>, tensor<double, c, dim> > qf_input{};

    for (int i = 0; i < c; i++) {
      for (int dy = tidy; dy < n; dy += q) {
        for (int qx = tidx; qx < q; qx += q) {
          double sum[2]{};
          for (int dx = 0; dx < n; dx++) {
            sum[0] += B(qx, dx) * X(i, dy, dx);
            sum[1] += G(qx, dx) * X(i, dy, dx);
          }
          A(0, dy, qx) = sum[0];
          A(1, dy, qx) = sum[1];
        }
      }
      __syncthreads();

      for (int qy = tidy; qy < q; qy += q) {
        for (int qx = tidx; qx < q; qx += q) {
          for (int dy = 0; dy < n; dy++) {
            get<0>(qf_input)[i] += B(qy, dy) * A(0, dy, qx);
            get<1>(qf_input)[i][0] += B(qy, dy) * A(1, dy, qx);
            get<1>(qf_input)[i][1] += G(qy, dy) * A(0, dy, qx);
          }
        }
      }
    }

    get<1>(qf_input) = dot(get<1>(qf_input), inv(J));

    return qf_input;
  }

  template <typename T1, typename T2, int q>
  static SERAC_DEVICE void integrate(tuple<T1, T2>& response, const tensor<double, dim, dim>& J,
                                     const TensorProductQuadratureRule<q>& rule, cache_type<q>& A, dof_type& residual)
  {
    int tidx = threadIdx.x % q;
    int tidy = threadIdx.x / q;

    static constexpr auto points1D  = GaussLegendreNodes<q>();
    static constexpr auto weights1D = GaussLegendreWeights<q>();
    static constexpr auto B_        = [=]() {
      tensor<double, q, n> B{};
      for (int i = 0; i < q; i++) {
        B[i] = GaussLobattoInterpolation<n>(points1D[i]);
      }
      return B;
    }();

    static constexpr auto G_ = [=]() {
      tensor<double, q, n> G{};
      for (int i = 0; i < q; i++) {
        G[i] = GaussLobattoInterpolationDerivative<n>(points1D[i]);
      }
      return G;
    }();

    __shared__ tensor<double, q, n> B;
    __shared__ tensor<double, q, n> G;
    for (int entry = threadIdx.x; entry < n * q; entry += q * q) {
      int i   = entry % n;
      int j   = entry / n;
      B(j, i) = B_(j, i);
      G(j, i) = G_(j, i);
    }
    __syncthreads();

    auto dv = det(J) * weights1D[tidx] * weights1D[tidy];

    get<0>(response) = get<0>(response) * dv;
    get<1>(response) = dot(get<1>(response), inv(transpose(J))) * dv;

    for (int i = 0; i < c; i++) {
      // this first contraction is performed a little differently, since `response` is not
      // in shared memory, so each thread can only access its own values
      for (int qy = tidy; qy < q; qy += q) {
        for (int dx = tidx; dx < n; dx += q) {
          A(0, dx, qy) = 0.0;
          A(1, dx, qy) = 0.0;
        }
      }
      __syncthreads();

      for (int offset = 0; offset < n; offset++) {
        int  dx  = (tidx + offset) % n;
        auto sum = B(tidx, dx) * get<0>(response)(i) + G(tidx, dx) * get<1>(response)(i, 0);
        atomicAdd(&A(0, dx, tidy), sum);
        atomicAdd(&A(1, dx, tidy), B(tidx, dx) * get<1>(response)(i, 1));
      }
      __syncthreads();

      for (int dy = tidy; dy < n; dy += q) {
        for (int dx = tidx; dx < n; dx += q) {
          double sum = 0.0;
          for (int qy = 0; qy < q; qy++) {
            sum += B(qy, dy) * A(0, dx, qy);
            sum += G(qy, dy) * A(1, dx, qy);
          }
          residual(i, dy, dx) += sum;
        }
      }
    }
  }

#endif

};
/// @endcond

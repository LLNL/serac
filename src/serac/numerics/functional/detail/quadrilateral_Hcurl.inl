// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file quadrilateral_Hcurl.inl
 *
 * @brief Specialization of finite_element for Hcurl on quadrilateral geometry
 */

// this specialization defines shape functions (and their curls) that
// interpolate at Gauss-Lobatto nodes for closed intervals, and Gauss-Legendre
// nodes for open intervals.
//
// note 1: mfem assumes the parent element domain is [0,1]x[0,1]
// note 2: dofs are numbered by direction and then lexicographically in space.
//         see below
// note 3: since this is a 2D element type, the "curl" returned is just the out-of-plane component,
//         rather than 3D vector along the z-direction.
// for additional information on the finite_element concept requirements, see finite_element.hpp
/// @cond
template <int p>
struct finite_element<Geometry::Quadrilateral, Hcurl<p> > {
  static constexpr auto geometry   = Geometry::Quadrilateral;
  static constexpr auto family     = Family::HCURL;
  static constexpr int  dim        = 2;
  static constexpr int  n          = (p + 1);
  static constexpr int  ndof       = 2 * p * (p + 1);
  static constexpr int  components = 1;

  using residual_type =
      typename std::conditional<components == 1, tensor<double, ndof>, tensor<double, ndof, components> >::type;

  // this is how mfem provides the data to us for these elements
  // if, instead, it was stored as simply tensor< double, 2, p + 1, p >,
  // the interpolation/integrate implementation would be considerably shorter
  struct dof_type {
    tensor<double, p + 1, p> x;
    tensor<double, p, p + 1> y;
  };

  /**
   * @brief this type is used when calling the batched interpolate/integrate
   *        routines, to provide memory for calculating intermediates
   */
  template <int q>
  using cache_type = tensor<double, p + 1, q>;

  template <int q>
  using cpu_batched_values_type = tensor<tensor<double, 2>, q, q>;

  template <int q>
  using cpu_batched_derivatives_type = tensor<double, q, q>;

  static constexpr auto directions = [] {
    int dof_per_direction = p * (p + 1);

    tensor<double, ndof, dim> directions{};
    for (int i = 0; i < dof_per_direction; i++) {
      directions[i + 0 * dof_per_direction] = {1.0, 0.0};
      directions[i + 1 * dof_per_direction] = {0.0, 1.0};
    }
    return directions;
  }();

  static constexpr auto nodes = [] {
    auto legendre_nodes = GaussLegendreNodes<p>();
    auto lobatto_nodes  = GaussLobattoNodes<p + 1>();

    tensor<double, ndof, dim> nodes{};

    int count = 0;
    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p; i++) {
        nodes[count++] = {legendre_nodes[i], lobatto_nodes[j]};
      }
    }

    for (int j = 0; j < p; j++) {
      for (int i = 0; i < p + 1; i++) {
        nodes[count++] = {lobatto_nodes[i], legendre_nodes[j]};
      }
    }

    return nodes;
  }();

  /*

    interpolate nodes/directions and their associated numbering:

                   linear

    o-----→-----o         o-----1-----o
    |           |         |           |
    |           |         |           |
    ↑           ↑         2           3
    |           |         |           |
    |           |         |           |
    o-----→-----o         o-----0-----o


                 quadratic

    o---→---→---o         o---4---5---o
    |           |         |           |
    ↑     ↑     ↑         9    10    11
    |   →   →   |         |   2   3   |
    ↑     ↑     ↑         6     7     8
    |           |         |           |
    o---→---→---o         o---0---1---o


                   cubic

    o--→--→--→--o         o--9-10-11--o
    ↑   ↑   ↑   ↑         20  21 22  23
    |  →  →  →  |         |  6  7  8  |
    ↑   ↑   ↑   ↑         16  17 18  19
    |  →  →  →  |         |  3  4  5  |
    ↑   ↑   ↑   ↑         12  13 14  15
    o--→--→--→--o         o--0--1--2--o

  */

  SERAC_HOST_DEVICE static constexpr tensor<double, ndof, dim> shape_functions(tensor<double, dim> xi)
  {
    int                       count = 0;
    tensor<double, ndof, dim> N{};

    // do all the x-facing nodes first
    tensor<double, p + 1> N_closed = GaussLobattoInterpolation<p + 1>(xi[1]);
    tensor<double, p>     N_open   = GaussLegendreInterpolation<p>(xi[0]);
    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p; i++) {
        N[count++] = {N_open[i] * N_closed[j], 0.0};
      }
    }

    // then all the y-facing nodes
    N_closed = GaussLobattoInterpolation<p + 1>(xi[0]);
    N_open   = GaussLegendreInterpolation<p>(xi[1]);
    for (int j = 0; j < p; j++) {
      for (int i = 0; i < p + 1; i++) {
        N[count++] = {0.0, N_closed[i] * N_open[j]};
      }
    }
    return N;
  }

  // the curl of a 2D vector field is entirely out-of-plane, so we return just that component
  SERAC_HOST_DEVICE static constexpr tensor<double, ndof> shape_function_curl(tensor<double, dim> xi)
  {
    int                  count = 0;
    tensor<double, ndof> curl_z{};

    // do all the x-facing nodes first
    tensor<double, p + 1> dN_closed = GaussLobattoInterpolationDerivative<p + 1>(xi[1]);
    tensor<double, p>     N_open    = GaussLegendreInterpolation<p>(xi[0]);
    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p; i++) {
        curl_z[count++] = -N_open[i] * dN_closed[j];
      }
    }

    // then all the y-facing nodes
    dN_closed = GaussLobattoInterpolationDerivative<p + 1>(xi[0]);
    N_open    = GaussLegendreInterpolation<p>(xi[1]);
    for (int j = 0; j < p; j++) {
      for (int i = 0; i < p + 1; i++) {
        curl_z[count++] = dN_closed[i] * N_open[j];
      }
    }

    return curl_z;
  }

  template <int q>
  static auto interpolate(const dof_type& element_values, const tensor<double, dim, dim, q, q>& jacobians,
                          const TensorProductQuadratureRule<q>&)
  {
    auto xi = GaussLegendreNodes<q>();

    tensor<double, q, p>     B1;
    tensor<double, q, p + 1> B2;
    tensor<double, q, p + 1> G2;
    for (int i = 0; i < q; i++) {
      B1[i] = GaussLegendreInterpolation<p>(xi[i]);
      B2[i] = GaussLobattoInterpolation<p + 1>(xi[i]);
      G2[i] = GaussLobattoInterpolationDerivative<p + 1>(xi[i]);
    }

    cache_type<q> A;

    serac::tuple<cpu_batched_values_type<q>, cpu_batched_derivatives_type<q> > values_and_derivatives{};

    /////////////////////////////////
    ////////// X-component //////////
    /////////////////////////////////
    for (int j = 0; j < p + 1; j++) {
      for (int qx = 0; qx < q; qx++) {
        double sum = 0.0;
        for (int i = 0; i < p; i++) {
          sum += B1(qx, i) * element_values.x(j, i);
        }
        A(j, qx) = sum;
      }
    }

    for (int qy = 0; qy < q; qy++) {
      for (int qx = 0; qx < q; qx++) {
        double sum[2]{};
        for (int j = 0; j < (p + 1); j++) {
          sum[0] += B2(qy, j) * A(j, qx);
          sum[1] += G2(qy, j) * A(j, qx);
        }
        serac::get<0>(values_and_derivatives)(qy, qx)[0] += sum[0];
        serac::get<1>(values_and_derivatives)(qy, qx) -= sum[1];
      }
    }

    /////////////////////////////////
    ////////// Y-component //////////
    /////////////////////////////////
    for (int i = 0; i < p + 1; i++) {
      for (int qy = 0; qy < q; qy++) {
        double sum = 0.0;
        for (int j = 0; j < p; j++) {
          sum += B1(qy, j) * element_values.y(j, i);
        }
        A(i, qy) = sum;
      }
    }

    for (int qy = 0; qy < q; qy++) {
      for (int qx = 0; qx < q; qx++) {
        double sum[3]{};
        for (int i = 0; i < (p + 1); i++) {
          sum[0] += B2(qx, i) * A(i, qy);
          sum[1] += G2(qx, i) * A(i, qy);
        }
        serac::get<0>(values_and_derivatives)(qy, qx)[1] += sum[0];
        serac::get<1>(values_and_derivatives)(qy, qx) += sum[1];
      }
    }

    // apply covariant Piola transformation to go
    // from parent element -> physical element
    for (int qy = 0; qy < q; qy++) {
      for (int qx = 0; qx < q; qx++) {
        tensor<double, dim, dim> J_T;
        for (int r = 0; r < dim; r++) {
          for (int c = 0; c < dim; c++) {
            J_T[r][c] = jacobians(r, c, qy, qx);
          }
        }
        auto detJ  = det(J_T);
        auto value = serac::get<0>(values_and_derivatives)(qy, qx);
        auto curl  = serac::get<1>(values_and_derivatives)(qy, qx);

        serac::get<0>(values_and_derivatives)(qy, qx) = linear_solve(J_T, value);
        serac::get<1>(values_and_derivatives)(qy, qx) = curl / detJ;
      }
    }

    return values_and_derivatives;
  }

  template <int q>
  static void integrate(cpu_batched_values_type<q>& sources, cpu_batched_derivatives_type<q>& fluxes,
                        const tensor<double, dim, dim, q, q>& jacobians, const TensorProductQuadratureRule<q>&,
                        dof_type&                             element_residual)
  {
    auto xi        = GaussLegendreNodes<q>();
    auto weights1D = GaussLegendreWeights<q>();

    tensor<double, q, p>     B1;
    tensor<double, q, p + 1> B2;
    tensor<double, q, p + 1> G2;
    for (int i = 0; i < q; i++) {
      B1[i] = GaussLegendreInterpolation<p>(xi[i]);
      B2[i] = GaussLobattoInterpolation<p + 1>(xi[i]);
      G2[i] = GaussLobattoInterpolationDerivative<p + 1>(xi[i]);
    }

    // transform the source and flux terms from values on the physical element,
    // to values on the parent element. Also, the source/flux values are scaled
    // according to the weight of their quadrature point, so that when we add them
    // together, it approximates the integral over the element
    for (int qy = 0; qy < q; qy++) {
      for (int qx = 0; qx < q; qx++) {
        tensor<double, dim, dim> J;
        for (int r = 0; r < dim; r++) {
          for (int c = 0; c < dim; c++) {
            J[r][c] = jacobians(c, r, qy, qx);
          }
        }
        auto detJ       = serac::det(J);
        auto dv         = detJ * weights1D[qx] * weights1D[qy];
        sources(qy, qx) = linear_solve(J, sources(qy, qx)) * dv;
        fluxes(qy, qx)  = fluxes(qy, qx) * (dv / detJ);
      }
    }

    cache_type<q> A;

    /////////////////////////////////
    ////////// X-component //////////
    /////////////////////////////////
    for (int j = 0; j < (p + 1); j++) {
      for (int qx = 0; qx < q; qx++) {
        double sum = 0.0;
        for (int qy = 0; qy < q; qy++) {
          sum += B2(qy, j) * sources(qy, qx)[0] - G2(qy, j) * fluxes(qy, qx);
        }
        A(j, qx) = sum;
      }
    }

    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p; i++) {
        double sum = 0.0;
        for (int qx = 0; qx < q; qx++) {
          sum += B1(qx, i) * A(j, qx);
        }
        element_residual.x(j, i) += sum;
      }
    }

    /////////////////////////////////
    ////////// Y-component //////////
    /////////////////////////////////
    for (int i = 0; i < (p + 1); i++) {
      for (int qy = 0; qy < q; qy++) {
        double sum = 0.0;
        for (int qx = 0; qx < q; qx++) {
          sum += B2(qx, i) * sources(qy, qx)[1] + G2(qx, i) * fluxes(qy, qx);
        }
        A(i, qy) = sum;
      }
    }

    for (int j = 0; j < p; j++) {
      for (int i = 0; i < p + 1; i++) {
        double sum = 0.0;
        for (int qy = 0; qy < q; qy++) {
          sum += B1(qy, j) * A(i, qy);
        }
        element_residual.y(j, i) += sum;
      }
    }
  }

  template <int q>
  static SERAC_DEVICE auto interpolate(const dof_type& element_values, const tensor<double, dim, dim>& J,
                                       const TensorProductQuadratureRule<q>& rule, cache_type<q>& A)
  {
    int tidx = threadIdx.x % q;
    int tidy = threadIdx.x / q;

    static constexpr auto points1D = GaussLegendreNodes<q>();

    static constexpr auto B1_ = [=]() {
      tensor<double, q, p> B1{};
      for (int i = 0; i < q; i++) {
        B1[i] = GaussLegendreInterpolation<p>(points1D[i]);
      }
      return B1;
    }();

    static constexpr auto B2_ = [=]() {
      tensor<double, q, n> B2{};
      for (int i = 0; i < q; i++) {
        B2[i] = GaussLobattoInterpolation<n>(points1D[i]);
      }
      return B2;
    }();

    static constexpr auto G2_ = [=]() {
      tensor<double, q, n> G2{};
      for (int i = 0; i < q; i++) {
        G2[i] = GaussLobattoInterpolationDerivative<n>(points1D[i]);
      }
      return G2;
    }();

    __shared__ tensor<double, q, p> B1;
    __shared__ tensor<double, q, n> B2;
    __shared__ tensor<double, q, n> G2;
    for (int entry = threadIdx.x; entry < n * q; entry += q * q) {
      int i = entry % n;
      int j = entry / n;
      if (i < p) {
        B1(j, i) = B1_(j, i);
      }
      B2(j, i) = B2_(j, i);
      G2(j, i) = G2_(j, i);
    }
    __syncthreads();

    tuple<tensor<double, dim>, double> qf_input{};

    /////////////////////////////////
    ////////// X-component //////////
    /////////////////////////////////
    for (int dy = tidy; dy < p + 1; dy += q) {
      for (int qx = tidx; qx < q; qx += q) {
        double sum = 0.0;
        for (int dx = 0; dx < p; dx++) {
          sum += B1(qx, dx) * element_values.x(dy, dx);
        }
        A(dy, qx) = sum;
      }
    }
    __syncthreads();

    for (int qy = tidy; qy < q; qy += q) {
      for (int qx = tidx; qx < q; qx += q) {
        double sum[2]{};
        for (int dy = 0; dy < (p + 1); dy++) {
          sum[0] += B2(qy, dy) * A(dy, qx);
          sum[1] += G2(qy, dy) * A(dy, qx);
        }
        serac::get<0>(qf_input)[0] += sum[0];
        serac::get<1>(qf_input) -= sum[1];
      }
    }
    __syncthreads();

    /////////////////////////////////
    ////////// Y-component //////////
    /////////////////////////////////
    for (int dx = tidx; dx < p + 1; dx += q) {
      for (int qy = tidy; qy < q; qy += q) {
        double sum = 0.0;
        for (int dy = 0; dy < p; dy++) {
          sum += B1(qy, dy) * element_values.y(dy, dx);
        }
        A(dx, qy) = sum;
      }
    }
    __syncthreads();

    for (int qy = tidy; qy < q; qy += q) {
      for (int qx = tidx; qx < q; qx += q) {
        double sum[3]{};
        for (int dx = 0; dx < (p + 1); dx++) {
          sum[0] += B2(qx, dx) * A(dx, qy);
          sum[1] += G2(qx, dx) * A(dx, qy);
        }
        serac::get<0>(qf_input)[1] += sum[0];
        serac::get<1>(qf_input) += sum[1];
      }
    }

    // apply covariant Piola transformation to go
    // from parent element -> physical element
    serac::get<0>(qf_input) = linear_solve(transpose(J), serac::get<0>(qf_input));
    serac::get<1>(qf_input) = serac::get<1>(qf_input) / det(J);

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

    static constexpr auto B1_ = [=]() {
      tensor<double, q, p> B1{};
      for (int i = 0; i < q; i++) {
        B1[i] = GaussLegendreInterpolation<p>(points1D[i]);
      }
      return B1;
    }();

    static constexpr auto B2_ = [=]() {
      tensor<double, q, n> B2{};
      for (int i = 0; i < q; i++) {
        B2[i] = GaussLobattoInterpolation<n>(points1D[i]);
      }
      return B2;
    }();

    static constexpr auto G2_ = [=]() {
      tensor<double, q, n> G2{};
      for (int i = 0; i < q; i++) {
        G2[i] = GaussLobattoInterpolationDerivative<n>(points1D[i]);
      }
      return G2;
    }();

    __shared__ tensor<double, q, p> B1;
    __shared__ tensor<double, q, n> B2;
    __shared__ tensor<double, q, n> G2;
    for (int entry = threadIdx.x; entry < n * q; entry += q * q) {
      int i = entry % n;
      int j = entry / n;
      if (i < p) {
        B1(j, i) = B1_(j, i);
      }
      B2(j, i) = B2_(j, i);
      G2(j, i) = G2_(j, i);
    }
    __syncthreads();

    // transform the source and flux terms from values on the physical element,
    // to values on the parent element. Also, the source/flux values are scaled
    // according to the weight of their quadrature point, so that when we add them
    // together, it approximates the integral over the element
    auto detJ = det(J);
    auto dv   = detJ * weights1D[tidx] * weights1D[tidy];

    get<0>(response) = linear_solve(J, get<0>(response)) * dv;
    get<1>(response) = get<1>(response) * (dv / detJ);

    /////////////////////////////////
    ////////// X-component //////////
    /////////////////////////////////

    // this first contraction is performed a little differently, since `response` is not
    // in shared memory, so each thread can only access its own values
    for (int qy = tidy; qy < q; qy += q) {
      for (int dx = tidx; dx < n; dx += q) {
        A(dx, qy) = 0.0;
      }
    }
    __syncthreads();

    for (int offset = 0; offset < n; offset++) {
      int  dy  = (tidy + offset) % n;
      auto sum = B2(tidy, dy) * get<0>(response)[0] - G2(tidy, dy) * get<1>(response);
      atomicAdd(&A(dy, tidx), sum);
    }
    __syncthreads();

    for (int dy = tidy; dy < p + 1; dy += q) {
      for (int dx = tidx; dx < p; dx += q) {
        double sum = 0.0;
        for (int qx = 0; qx < q; qx++) {
          sum += B1(qx, dx) * A(dy, qx);
        }
        residual.x(dy, dx) += sum;
      }
    }
    __syncthreads();

    /////////////////////////////////
    ////////// Y-component //////////
    /////////////////////////////////

    // this first contraction is performed a little differently, since `response` is not
    // in shared memory, so each thread can only access its own values
    for (int qy = tidy; qy < q; qy += q) {
      for (int dx = tidx; dx < n; dx += q) {
        A(dx, qy) = 0.0;
      }
    }
    __syncthreads();

    for (int offset = 0; offset < n; offset++) {
      int  dx  = (tidx + offset) % n;
      auto sum = B2(tidx, dx) * get<0>(response)[1] + G2(tidx, dx) * get<1>(response);
      atomicAdd(&A(dx, tidy), sum);
    }
    __syncthreads();

    for (int dy = tidy; dy < p; dy += q) {
      for (int dx = tidx; dx < p + 1; dx += q) {
        double sum = 0.0;
        for (int qy = 0; qy < q; qy++) {
          sum += B1(qy, dy) * A(dx, qy);
        }
        residual.y(dy, dx) += sum;
      }
    }
  }
};
/// @endcond

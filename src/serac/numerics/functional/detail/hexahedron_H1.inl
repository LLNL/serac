// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file hexahedron_H1.inl
 *
 * @brief Specialization of finite_element for H1 on hexahedron geometry
 */

// this specialization defines shape functions (and their gradients) that
// interpolate at Gauss-Lobatto nodes for the appropriate polynomial order
//
// note: mfem assumes the parent element domain is [0,1]x[0,1]x[0,1]
// for additional information on the finite_element concept requirements, see finite_element.hpp
/// @cond
template <int p, int c>
struct finite_element<Geometry::Hexahedron, H1<p, c> > {
  static constexpr auto geometry   = Geometry::Hexahedron;
  static constexpr auto family     = Family::H1;
  static constexpr int  components = c;
  static constexpr int  dim        = 3;
  static constexpr int  n          = (p + 1);
  static constexpr int  ndof       = (p + 1) * (p + 1) * (p + 1);
  static constexpr int  order      = p;

  // TODO: remove this
  using residual_type =
      typename std::conditional<components == 1, tensor<double, ndof>, tensor<double, ndof, components> >::type;

  using dof_type = tensor<double, c, p + 1, p + 1, p + 1>;

  using value_type = typename std::conditional<components == 1, double, tensor<double, components> >::type;
  using derivative_type = typename std::conditional<components == 1, tensor<double, dim>, tensor<double, components, dim> >::type;
  using qf_input_type = tuple< value_type, derivative_type >;

  /**
   * @brief this type is used when calling the batched interpolate/integrate
   *        routines, to provide memory for calculating intermediates
   */
  template <int q>
  struct cache_type {
    tensor<double, 2, n, n, q> A1;
    tensor<double, 3, n, q, q> A2;
  };

  template <typename T>
  using simd_dof_type = tensor<tensor<T, c>, p + 1, p + 1, p + 1>;

  template <typename T, int q>
  struct simd_cache_type {
    tensor<T, 2, n, n, q> A1;
    tensor<T, 3, n, q, q> A2;
  };

  SERAC_HOST_DEVICE static constexpr tensor<double, ndof> shape_functions(tensor<double, dim> xi)
  {
    auto N_xi   = GaussLobattoInterpolation<p + 1>(xi[0]);
    auto N_eta  = GaussLobattoInterpolation<p + 1>(xi[1]);
    auto N_zeta = GaussLobattoInterpolation<p + 1>(xi[2]);

    int count = 0;

    tensor<double, ndof> N{};
    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p + 1; i++) {
          N[count++] = N_xi[i] * N_eta[j] * N_zeta[k];
        }
      }
    }
    return N;
  }

  SERAC_HOST_DEVICE static constexpr tensor<double, ndof, dim> shape_function_gradients(tensor<double, dim> xi)
  {
    auto N_xi    = GaussLobattoInterpolation<p + 1>(xi[0]);
    auto N_eta   = GaussLobattoInterpolation<p + 1>(xi[1]);
    auto N_zeta  = GaussLobattoInterpolation<p + 1>(xi[2]);
    auto dN_xi   = GaussLobattoInterpolationDerivative<p + 1>(xi[0]);
    auto dN_eta  = GaussLobattoInterpolationDerivative<p + 1>(xi[1]);
    auto dN_zeta = GaussLobattoInterpolationDerivative<p + 1>(xi[2]);

    int count = 0;

    // clang-format off
    tensor<double, ndof, dim> dN{};
    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p + 1; i++) {
          dN[count++] = {
            dN_xi[i] *  N_eta[j] *  N_zeta[k], 
             N_xi[i] * dN_eta[j] *  N_zeta[k],
             N_xi[i] *  N_eta[j] * dN_zeta[k]
          };
        }
      }
    }
    return dN;
    // clang-format on
  }

  template <int q>
  static auto interpolate(const dof_type& X, const tensor<double, dim, dim, q * q * q>& jacobians,
                          const TensorProductQuadratureRule<q>&)
  {
    // we want to compute the following:
    //
    // X_q(u, v, w) := (B(u, i) * B(v, j) * B(w, k)) * X_e(i, j, k)
    //
    // where
    //   X_q(u, v, w) are the quadrature-point values at position {u, v, w},
    //   B(u, i) is the i^{th} 1D interpolation/differentiation (shape) function,
    //           evaluated at the u^{th} 1D quadrature point, and
    //   X_e(i, j, k) are the values at node {i, j, k} to be interpolated
    //
    // this algorithm carries out the above calculation in 3 steps:
    //
    // A1(dz, dy, qx)  := B(qx, dx) * X_e(dz, dy, dx)
    // A2(dz, qy, qx)  := B(qy, dy) * A1(dz, dy, qx)
    // X_q(qz, qy, qx) := B(qz, dz) * A2(dz, qy, qx)

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

    cache_type<q> cache;

    tensor< double, c, q, q, q> value{};
    tensor< double, c, dim, q, q, q> gradient{};

    for (int i = 0; i < c; i++) {
      cache.A1[0] = contract<2, 1>(X[i], B);
      cache.A1[1] = contract<2, 1>(X[i], G);

      cache.A2[0] = contract<1, 1>(cache.A1[0], B);
      cache.A2[1] = contract<1, 1>(cache.A1[1], B);
      cache.A2[2] = contract<1, 1>(cache.A1[0], G);

      value(i)      = contract<0, 1>(cache.A2[0], B);
      gradient(i,0) = contract<0, 1>(cache.A2[1], B);
      gradient(i,1) = contract<0, 1>(cache.A2[2], B);
      gradient(i,2) = contract<0, 1>(cache.A2[0], G);
    }

    constexpr int VALUE = 0, GRADIENT = 1;

    union {
      tensor< qf_input_type, q * q * q > a;
      tensor< tuple < tensor< double, c >, tensor< double, c, 3 > >, q * q * q > b;
    } output;

    int k = 0;
    for (int qz = 0; qz < q; qz++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          tensor<double, dim, dim> J;
          for (int row = 0; row < dim; row++) {
            for (int col = 0; col < dim; col++) {
              J[row][col] = jacobians(col, row, k);
            }
          }

          for (int i = 0; i < c; i++) {
            get<VALUE>(output.b[k])[i] = value(i, qz, qy, qx);
            for (int j = 0; j < dim; j++) {
              get<GRADIENT>(output.b[k])[i][j] = gradient(i, j, qz, qy, qx);
            }
          }

          get<GRADIENT>(output.b[k]) = dot(get<GRADIENT>(output.b[k]), inv(J));

          k++;
        }
      }
    }
 
    return output.a;
  }

  template <typename source_type, typename flux_type, int q>
  static void integrate(tensor< tuple< source_type, flux_type >, q * q * q > & qf_output,
                        const tensor<double, dim, dim, q * q * q>& jacobians, const TensorProductQuadratureRule<q>&,
                        dof_type&                                element_residual)
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

    tensor< double, c, q, q, q> source{};
    tensor< double, c, dim, q, q, q> flux{};

    constexpr int SOURCE = 0, FLUX = 1;

    int k = 0;
    for (int qz = 0; qz < q; qz++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          tensor<double, dim, dim> J_T;
          for (int row = 0; row < dim; row++) {
            for (int col = 0; col < dim; col++) {
              J_T[row][col] = jacobians(row, col, k);
            }
          }
          auto dv = det(J_T) * weights1D[qx] * weights1D[qy] * weights1D[qz];

          tensor< double, c > s{get<SOURCE>(qf_output[k]) * dv};
          tensor< double, c, dim > f{dot(get<FLUX>(qf_output[k]), inv(J_T)) * dv};

          for (int i = 0; i < c; i++) {
            source(i, qz, qy, qx) = s[i];
            for (int j = 0; j < dim; j++) {
              flux(i, j, qz, qy, qx) = f[i][j];
            }
          }
          k++;
        }
      }
    }

    cache_type<q> cache{};

    for (int i = 0; i < c; i++) {
      cache.A2[0] = contract< 2, 0 >(source[i], B) + contract< 2, 0 >(flux(i, 0), G);
      cache.A2[1] = contract< 2, 0 >(flux(i, 1), B);
      cache.A2[2] = contract< 2, 0 >(flux(i, 2), B);

      cache.A1[0] = contract< 1, 0 >(cache.A2[0], B) + contract< 1, 0 >(cache.A2[1], G);
      cache.A1[1] = contract< 1, 0 >(cache.A2[2], B);

      element_residual(i) += contract< 0, 0 >(cache.A1[0], B) + contract< 0, 0 >(cache.A1[1], G);
    }
  }

#if defined(__CUDACC__)

  template <int q>
  static SERAC_DEVICE auto interpolate(const dof_type& X, const tensor<double, dim, dim>& J,
                                       const TensorProductQuadratureRule<q>& rule, cache_type<q>& cache)
  {
    // we want to compute the following:
    //
    // X_q(u, v, w) := (B(u, i) * B(v, j) * B(w, k)) * X_e(i, j, k)
    //
    // where
    //   X_q(u, v, w) are the quadrature-point values at position {u, v, w},
    //   B(u, i) is the i^{th} 1D interpolation/differentiation (shape) function,
    //           evaluated at the u^{th} 1D quadrature point, and
    //   X_e(i, j, k) are the values at node {i, j, k} to be interpolated
    //
    // this algorithm carries out the above calculation in 3 steps:
    //
    // A1(dz, dy, qx)  := B(qx, dx) * X_e(dz, dy, dx)
    // A2(dz, qy, qx)  := B(qy, dy) * A1(dz, dy, qx)
    // X_q(qz, qy, qx) := B(qz, dz) * A2(dz, qy, qx)

    int tidx = threadIdx.x % q;
    int tidy = (threadIdx.x % (q * q)) / q;
    int tidz = threadIdx.x / (q * q);

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
    for (int entry = threadIdx.x; entry < n * q; entry += q * q * q) {
      int i   = entry % n;
      int j   = entry / n;
      B(j, i) = B_(j, i);
      G(j, i) = G_(j, i);
    }
    __syncthreads();

    tuple<tensor<double, c>, tensor<double, c, 3> > qf_input{};

    for (int i = 0; i < c; i++) {
      for (int dz = tidz; dz < n; dz += q) {
        for (int dy = tidy; dy < n; dy += q) {
          for (int qx = tidx; qx < q; qx += q) {
            double sum[2]{};
            for (int dx = 0; dx < n; dx++) {
              sum[0] += B(qx, dx) * X(i, dz, dy, dx);
              sum[1] += G(qx, dx) * X(i, dz, dy, dx);
            }
            cache.A1(0, dz, dy, qx) = sum[0];
            cache.A1(1, dz, dy, qx) = sum[1];
          }
        }
      }
      __syncthreads();

      for (int dz = tidz; dz < n; dz += q) {
        for (int qy = tidy; qy < q; qy += q) {
          for (int qx = tidx; qx < q; qx += q) {
            double sum[3]{};
            for (int dy = 0; dy < n; dy++) {
              sum[0] += B(qy, dy) * cache.A1(0, dz, dy, qx);
              sum[1] += B(qy, dy) * cache.A1(1, dz, dy, qx);
              sum[2] += G(qy, dy) * cache.A1(0, dz, dy, qx);
            }
            cache.A2(0, dz, qy, qx) = sum[0];
            cache.A2(1, dz, qy, qx) = sum[1];
            cache.A2(2, dz, qy, qx) = sum[2];
          }
        }
      }
      __syncthreads();

      for (int qz = tidz; qz < q; qz += q) {
        for (int qy = tidy; qy < q; qy += q) {
          for (int qx = tidx; qx < q; qx += q) {
            for (int dz = 0; dz < n; dz++) {
              get<0>(qf_input)[i] += B(qz, dz) * cache.A2(0, dz, qy, qx);
              get<1>(qf_input)[i][0] += B(qz, dz) * cache.A2(1, dz, qy, qx);
              get<1>(qf_input)[i][1] += B(qz, dz) * cache.A2(2, dz, qy, qx);
              get<1>(qf_input)[i][2] += G(qz, dz) * cache.A2(0, dz, qy, qx);
            }
          }
        }
      }
    }

    get<1>(qf_input) = dot(get<1>(qf_input), inv(J));

    return qf_input;
  }

  template <typename T1, typename T2, int q>
  static SERAC_DEVICE void integrate(tuple<T1, T2>& response, const tensor<double, dim, dim>& J,
                                     const TensorProductQuadratureRule<q>& rule, cache_type<q>& cache,
                                     dof_type& residual)
  {
    int tidx = threadIdx.x % q;
    int tidy = (threadIdx.x % (q * q)) / q;
    int tidz = threadIdx.x / (q * q);

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
    for (int entry = threadIdx.x; entry < n * q; entry += q * q * q) {
      int i   = entry % n;
      int j   = entry / n;
      B(j, i) = B_(j, i);
      G(j, i) = G_(j, i);
    }
    __syncthreads();

    auto dv = det(J) * weights1D[tidx] * weights1D[tidy] * weights1D[tidz];

    get<0>(response) = get<0>(response) * dv;
    get<1>(response) = dot(get<1>(response), inv(transpose(J))) * dv;

    for (int i = 0; i < c; i++) {
      // this first contraction is performed a little differently, since `response` is not
      // in shared memory, so each thread can only access its own values
      for (int qz = tidz; qz < q; qz += q) {
        for (int qy = tidy; qy < q; qy += q) {
          for (int dx = tidx; dx < n; dx += q) {
            cache.A2(0, dx, qy, qz) = 0.0;
            cache.A2(1, dx, qy, qz) = 0.0;
            cache.A2(2, dx, qy, qz) = 0.0;
          }
        }
      }
      __syncthreads();

      for (int offset = 0; offset < n; offset++) {
        int  dx  = (tidx + offset) % n;
        auto sum = B(tidx, dx) * get<0>(response)(i) + G(tidx, dx) * get<1>(response)(i, 0);
        atomicAdd(&cache.A2(0, dx, tidz, tidy), sum);
        atomicAdd(&cache.A2(1, dx, tidz, tidy), B(tidx, dx) * get<1>(response)(i, 1));
        atomicAdd(&cache.A2(2, dx, tidz, tidy), B(tidx, dx) * get<1>(response)(i, 2));
      }
      __syncthreads();

      for (int qz = tidz; qz < q; qz += q) {
        for (int dy = tidy; dy < n; dy += q) {
          for (int dx = tidx; dx < n; dx += q) {
            double sum[2]{};
            for (int qy = 0; qy < q; qy++) {
              sum[0] += B(qy, dy) * cache.A2(0, dx, qz, qy);
              sum[0] += G(qy, dy) * cache.A2(1, dx, qz, qy);
              sum[1] += B(qy, dy) * cache.A2(2, dx, qz, qy);
            }
            cache.A1(0, qz, dy, dx) = sum[0];
            cache.A1(1, qz, dy, dx) = sum[1];
          }
        }
      }
      __syncthreads();

      for (int dz = tidz; dz < n; dz += q) {
        for (int dy = tidy; dy < n; dy += q) {
          for (int dx = tidx; dx < n; dx += q) {
            double sum = 0.0;
            for (int qz = 0; qz < q; qz++) {
              sum += B(qz, dz) * cache.A1(0, qz, dy, dx);
              sum += G(qz, dz) * cache.A1(1, qz, dy, dx);
            }
            residual(i, dz, dy, dx) += sum;
          }
        }
      }
    }
  }

#endif

};
/// @endcond

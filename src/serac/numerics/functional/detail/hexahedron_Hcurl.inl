// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file hexahedron_Hcurl.inl
 *
 * @brief Specialization of finite_element for Hcurl on hexahedron geometry
 */

// this specialization defines shape functions (and their curls) that
// interpolate at Gauss-Lobatto nodes for closed intervals, and Gauss-Legendre
// nodes for open intervals.
//
// note 1: mfem assumes the parent element domain is [0,1]x[0,1]x[0,1]
// note 2: dofs are numbered by direction and then lexicographically in space.
//         quadrilateral_hcurl.inl for more information
// for additional information on the finite_element concept requirements, see finite_element.hpp
/// @cond
template <int p>
struct finite_element<Geometry::Hexahedron, Hcurl<p>> {
  static constexpr auto geometry   = Geometry::Hexahedron;
  static constexpr auto family     = Family::HCURL;
  static constexpr int  dim        = 3;
  static constexpr int  ndof       = 3 * p * (p + 1) * (p + 1);
  static constexpr int  components = 1;

  // TODO: delete this in favor of dof_type
  using residual_type =
      typename std::conditional<components == 1, tensor<double, ndof>, tensor<double, ndof, components>>::type;

  // this is how mfem provides the data to us for these elements
  // if, instead, it was stored as simply tensor< double, 3, p + 1, p + 1, p >,
  // the interpolation/integrate implementation would be considerably shorter
  struct dof_type {
    tensor< double, p + 1, p + 1, p     > x;
    tensor< double, p + 1, p    , p + 1 > y;
    tensor< double, p    , p + 1, p + 1 > z;
  };

  /**
   * @brief this type is used when calling the batched interpolate/integrate
   *        routines, to provide memory for calculating intermediates
   */
  template < int q >
  struct cache_type {
    tensor<double, p + 1, p + 1, q> A1; 
    tensor<double, 2, p + 1, q, q> A2;
  };

  template < int q >
  using cpu_batched_values_type = tensor< tensor< double, 3 >, q, q, q >;

  template < int q >
  using cpu_batched_derivatives_type = tensor< tensor< double, 3 >, q, q, q >;

  static constexpr auto directions = [] {
    int dof_per_direction = p * (p + 1) * (p + 1);

    tensor<double, ndof, dim> directions{};
    for (int i = 0; i < dof_per_direction; i++) {
      directions[i + 0 * dof_per_direction] = {1.0, 0.0, 0.0};
      directions[i + 1 * dof_per_direction] = {0.0, 1.0, 0.0};
      directions[i + 2 * dof_per_direction] = {0.0, 0.0, 1.0};
    }
    return directions;
  }();

  static constexpr auto nodes = []() {
    auto legendre_nodes = GaussLegendreNodes<p>();
    auto lobatto_nodes  = GaussLobattoNodes<p + 1>();

    tensor<double, ndof, dim> nodes{};

    int count = 0;
    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p; i++) {
          nodes[count++] = {legendre_nodes[i], lobatto_nodes[j], lobatto_nodes[k]};
        }
      }
    }

    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p; j++) {
        for (int i = 0; i < p + 1; i++) {
          nodes[count++] = {lobatto_nodes[i], legendre_nodes[j], lobatto_nodes[k]};
        }
      }
    }

    for (int k = 0; k < p; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p + 1; i++) {
          nodes[count++] = {lobatto_nodes[i], lobatto_nodes[j], legendre_nodes[k]};
        }
      }
    }

    return nodes;
  }();



  SERAC_HOST_DEVICE static constexpr tensor<double, ndof, dim> shape_functions(tensor<double, dim> xi)
  {
    tensor<double, ndof, dim> N{};

    tensor<double, p> f[3] = {GaussLegendreInterpolation<p>(xi[0]), GaussLegendreInterpolation<p>(xi[1]),
                              GaussLegendreInterpolation<p>(xi[2])};

    tensor<double, p + 1> g[3] = {GaussLobattoInterpolation<p + 1>(xi[0]), GaussLobattoInterpolation<p + 1>(xi[1]),
                                  GaussLobattoInterpolation<p + 1>(xi[2])};

    int count = 0;

    // do all the x-facing nodes first
    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p; i++) {
          N[count++] = {f[0][i] * g[1][j] * g[2][k], 0.0, 0.0};
        }
      }
    }

    // then all the y-facing nodes
    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p; j++) {
        for (int i = 0; i < p + 1; i++) {
          N[count++] = {0.0, g[0][i] * f[1][j] * g[2][k], 0.0};
        }
      }
    }

    // then, finally, all the z-facing nodes
    for (int k = 0; k < p; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p + 1; i++) {
          N[count++] = {0.0, 0.0, g[0][i] * g[1][j] * f[2][k]};
        }
      }
    }

    return N;
  }

  SERAC_HOST_DEVICE static constexpr tensor<double, ndof, dim> shape_function_curl(tensor<double, dim> xi)
  {
    tensor<double, ndof, dim> curl{};

    tensor<double, p> f[3] = {GaussLegendreInterpolation<p>(xi[0]), GaussLegendreInterpolation<p>(xi[1]),
                              GaussLegendreInterpolation<p>(xi[2])};

    tensor<double, p + 1> g[3] = {GaussLobattoInterpolation<p + 1>(xi[0]), GaussLobattoInterpolation<p + 1>(xi[1]),
                                  GaussLobattoInterpolation<p + 1>(xi[2])};

    tensor<double, p + 1> dg[3] = {GaussLobattoInterpolationDerivative<p + 1>(xi[0]),
                                   GaussLobattoInterpolationDerivative<p + 1>(xi[1]),
                                   GaussLobattoInterpolationDerivative<p + 1>(xi[2])};

    int count = 0;

    // do all the x-facing nodes first
    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p; i++) {
          // curl({f(x) g(y) g(z), 0.0, 0.0}) == {0.0, f(x) g(y) g'(z), -f(x) g'(y) g(z)};
          curl[count++] = {0.0, f[0][i] * g[1][j] * dg[2][k], -f[0][i] * dg[1][j] * g[2][k]};
        }
      }
    }

    // then all the y-facing nodes
    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p; j++) {
        for (int i = 0; i < p + 1; i++) {
          // curl({0.0, g(x) f(y) g(z), 0.0}) == {-g(x) f(y) g'(z), 0.0, g'(x) f(y) g(z)};
          curl[count++] = {-g[0][i] * f[1][j] * dg[2][k], 0.0, dg[0][i] * f[1][j] * g[2][k]};
        }
      }
    }

    // then, finally, all the z-facing nodes
    for (int k = 0; k < p; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p + 1; i++) {
          // curl({0.0, 0.0, g(x) g(y) f(z)}) == {g(x) g'(y) f(z), -g'(x) g(y) f(z), 0.0};
          curl[count++] = {g[0][i] * dg[1][j] * f[2][k], -dg[0][i] * g[1][j] * f[2][k], 0.0};
        }
      }
    }

    return curl;
  }

  template < int q >
  static auto interpolate(const dof_type & element_values, 
                          const tensor < double, q, q, q, dim, dim > & jacobians,
                          const TensorProductQuadratureRule<q> &) {

    auto xi = GaussLegendreNodes<q>();

    tensor<double, q, p > B1;
    tensor<double, q, p+1 > B2;
    tensor<double, q, p+1 > G2;
    for (int i = 0; i < q; i++) {
      B1[i] = GaussLegendreInterpolation<p>(xi[i]);
      B2[i] = GaussLobattoInterpolation<p+1>(xi[i]);
      G2[i] = GaussLobattoInterpolationDerivative<p+1>(xi[i]);
    }

    cache_type<q> cache;

    serac::tuple < cpu_batched_values_type<q>, cpu_batched_derivatives_type<q> > values_and_derivatives{};

    /////////////////////////////////
    ////////// X-component //////////
    /////////////////////////////////
    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int qx = 0; qx < q; qx++) {
          double sum = 0.0;
          for (int i = 0; i < p; i++) {
            sum += B1(qx, i) * element_values.x(k, j, i);
          }
          cache.A1(k, j, qx) = sum;
        }
      }
    }

    for (int k = 0; k < p + 1; k++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          double sum[2]{};
          for (int j = 0; j < (p + 1); j++) {
            sum[0] += B2(qy, j) * cache.A1(k, j, qx);
            sum[1] += G2(qy, j) * cache.A1(k, j, qx);
          }
          cache.A2(0, k, qy, qx) = sum[0];
          cache.A2(1, k, qy, qx) = sum[1];
        }
      }
    }

    for (int qz = 0; qz < q; qz++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          double sum[3]{};
          for (int k = 0; k < (p + 1); k++) {
            sum[0] += B2(qz, k) * cache.A2(0, k, qy, qx);
            sum[1] += G2(qz, k) * cache.A2(0, k, qy, qx);
            sum[2] += B2(qz, k) * cache.A2(1, k, qy, qx);
          }
          serac::get<0>(values_and_derivatives)(qz, qy, qx)[0] += sum[0];
          serac::get<1>(values_and_derivatives)(qz, qy, qx)[1] += sum[1];
          serac::get<1>(values_and_derivatives)(qz, qy, qx)[2] -= sum[2];
        }
      }
    }

    /////////////////////////////////
    ////////// Y-component //////////
    /////////////////////////////////
    for (int k = 0; k < p + 1; k++) {
      for (int i = 0; i < p + 1; i++) {
        for (int qy = 0; qy < q; qy++) {
          double sum = 0.0;
          for (int j = 0; j < p; j++) {
            sum += B1(qy, j) * element_values.y(k, j, i);
          }
          cache.A1(k, i, qy) = sum;
        }
      }
    }

    for (int k = 0; k < p + 1; k++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          double sum[2]{};
          for (int i = 0; i < (p + 1); i++) {
            sum[0] += B2(qx, i) * cache.A1(k, i, qy);
            sum[1] += G2(qx, i) * cache.A1(k, i, qy);
          }
          cache.A2(0, k, qy, qx) = sum[0];
          cache.A2(1, k, qy, qx) = sum[1];
        }
      }
    }

    for (int qz = 0; qz < q; qz++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          double sum[3]{};
          for (int k = 0; k < (p + 1); k++) {
            sum[0] += B2(qz, k) * cache.A2(0, k, qy, qx);
            sum[1] += G2(qz, k) * cache.A2(0, k, qy, qx);
            sum[2] += B2(qz, k) * cache.A2(1, k, qy, qx);
          }
          serac::get<0>(values_and_derivatives)(qz, qy, qx)[1] += sum[0];
          serac::get<1>(values_and_derivatives)(qz, qy, qx)[2] += sum[2];
          serac::get<1>(values_and_derivatives)(qz, qy, qx)[0] -= sum[1];
        }
      }
    }

    /////////////////////////////////
    ////////// Z-component //////////
    /////////////////////////////////
    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p + 1; i++) {
        for (int qz = 0; qz < q; qz++) {
          double sum = 0.0;
          for (int k = 0; k < p; k++) {
            sum += B1(qz, k) * element_values.z(k, j, i);
          }
          cache.A1(j, i, qz) = sum;
        }
      }
    }

    for (int j = 0; j < p + 1; j++) {
      for (int qz = 0; qz < q; qz++) {
        for (int qx = 0; qx < q; qx++) {
          double sum[2]{};
          for (int i = 0; i < (p + 1); i++) {
            sum[0] += B2(qx, i) * cache.A1(j, i, qz);
            sum[1] += G2(qx, i) * cache.A1(j, i, qz);
          }
          cache.A2(0, j, qz, qx) = sum[0];
          cache.A2(1, j, qz, qx) = sum[1];
        }
      }
    }

    for (int qz = 0; qz < q; qz++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          double sum[3]{};
          for (int j = 0; j < (p + 1); j++) {
            sum[0] += B2(qy, j) * cache.A2(0, j, qz, qx);
            sum[1] += G2(qy, j) * cache.A2(0, j, qz, qx);
            sum[2] += B2(qy, j) * cache.A2(1, j, qz, qx);
          }
          serac::get<0>(values_and_derivatives)(qz, qy, qx)[2] += sum[0];
          serac::get<1>(values_and_derivatives)(qz, qy, qx)[0] += sum[1];
          serac::get<1>(values_and_derivatives)(qz, qy, qx)[1] -= sum[2];
        }
      }
    }

    // apply covariant Piola transformation to go
    // from parent element -> physical element
    for (int qz = 0; qz < q; qz++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          tensor< double, dim, dim > J_T;
          for (int row = 0; row < dim; row++) {
            for (int col = 0; col < dim; col++) {
              J_T[row][col] = jacobians(row, col, qz, qy, qx);
            }
          }
          auto detJ = det(J_T);
          auto value = serac::get<0>(values_and_derivatives)(qz, qy, qx);
          auto curl = serac::get<1>(values_and_derivatives)(qz, qy, qx);
          serac::get<0>(values_and_derivatives)(qz, qy, qx) = linear_solve(J_T, value);
          serac::get<1>(values_and_derivatives)(qz, qy, qx) = dot(curl, J_T) / detJ;
        }
      }
    }

    return values_and_derivatives;

  }

  template < int q >
  static void integrate(cpu_batched_values_type<q> & sources,
                        cpu_batched_derivatives_type<q> & fluxes,
                        const tensor < double, q, q, q, dim, dim > & jacobians,
                        const TensorProductQuadratureRule<q> &, 
                        dof_type & element_residual) {

    static constexpr auto xi = GaussLegendreNodes<q>();
    static constexpr auto weights1D = GaussLegendreWeights<q>();

    cache_type<q> cache{};
    tensor<double, q, p > B1;
    tensor<double, q, p+1 > B2;
    tensor<double, q, p+1 > G2;
    for (int i = 0; i < q; i++) {
      B1[i] = GaussLegendreInterpolation<p>(xi[i]);
      B2[i] = GaussLobattoInterpolation<p+1>(xi[i]);
      G2[i] = GaussLobattoInterpolationDerivative<p+1>(xi[i]);
    }

    // transform the source and flux terms from values on the physical element, 
    // to values on the parent element. Also, the source/flux values are scaled
    // according to the weight of their quadrature point, so that 
    for (int qz = 0; qz < q; qz++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          tensor< double, dim, dim > J;
          for (int row = 0; row < dim; row++) {
            for (int col = 0; col < dim; col++) {
              J[row][col] = jacobians(col, row, qz, qy, qx);
            }
          }
          auto detJ = serac::det(J);
          auto dv = detJ * weights1D[qx] * weights1D[qy] * weights1D[qz];
          sources(qz, qy, qx) = linear_solve(J, sources(qz, qy, qx)) * dv;
          fluxes(qz, qy, qx) = dot(fluxes(qz, qy, qx), J) * (dv / detJ);
        }
      }
    }

    /////////////////////////////////
    ////////// X-component //////////
    /////////////////////////////////
    for (int k = 0; k < p + 1; k++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          double sum[2]{};
          for (int qz = 0; qz < q; qz++) {
            sum[0] += B2(qz, k) * sources(qz, qy, qx)[0] + G2(qz, k) * fluxes(qz, qy, qx)[1];
            sum[1] -= B2(qz, k) * fluxes(qz, qy, qx)[2];
          }
          cache.A2(0, k, qy, qx) = sum[0];
          cache.A2(1, k, qy, qx) = sum[1];
        }
      }
    }

    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int qx = 0; qx < q; qx++) {
          double sum = 0.0;
          for (int qy = 0; qy < q; qy++) {
            sum += B2(qy, j) * cache.A2(0, k, qy, qx);
            sum += G2(qy, j) * cache.A2(1, k, qy, qx);
          }
          cache.A1(k, j, qx) = sum;
        }
      }
    }

    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p; i++) {
          double sum = 0.0;
          for (int qx = 0; qx < q; qx++) {
            sum += B1(qx, i) * cache.A1(k, j, qx);
          }
          element_residual.x(k, j, i) += sum;
        }
      }
    }

    /////////////////////////////////
    ////////// Y-component //////////
    /////////////////////////////////
    for (int k = 0; k < p + 1; k++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          double sum[2]{};
          for (int qz = 0; qz < q; qz++) {
            sum[0] += B2(qz, k) * sources(qz, qy, qx)[1] - G2(qz, k) * fluxes(qz, qy, qx)[0];
            sum[1] += B2(qz, k) * fluxes(qz, qy, qx)[2];
          }
          cache.A2(0, k, qy, qx) = sum[0];
          cache.A2(1, k, qy, qx) = sum[1];
        }
      }
    }

    for (int k = 0; k < p + 1; k++) {
      for (int i = 0; i < p + 1; i++) {
        for (int qy = 0; qy < q; qy++) {
          double sum = 0.0;
          for (int qx = 0; qx < q; qx++) {
            sum += B2(qx, i) * cache.A2(0, k, qy, qx);
            sum += G2(qx, i) * cache.A2(1, k, qy, qx);
          }
          cache.A1(k, i, qy) = sum;
        }
      }
    }

    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p; j++) {
        for (int i = 0; i < p + 1; i++) {
          double sum = 0.0;
          for (int qy = 0; qy < q; qy++) {
            sum += B1(qy, j) * cache.A1(k, i, qy);
          }
          element_residual.y(k, j, i) += sum;
        }
      }
    }

    /////////////////////////////////
    ////////// Z-component //////////
    /////////////////////////////////
    for (int i = 0; i < p + 1; i++) {
      for (int qz = 0; qz < q; qz++) {
        for (int qy = 0; qy < q; qy++) {
          double sum[2]{};
          for (int qx = 0; qx < q; qx++) {
            sum[0] += B2(qx, i) * sources(qz, qy, qx)[2] - G2(qx, i) * fluxes(qz, qy, qx)[1];
            sum[1] += B2(qx, i) * fluxes(qz, qy, qx)[0];
          }
          cache.A2(0, i, qz, qy) = sum[0];
          cache.A2(1, i, qz, qy) = sum[1];
        }
      }
    }

    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p + 1; i++) {
        for (int qz = 0; qz < q; qz++) {
          double sum = 0.0;
          for (int qy = 0; qy < q; qy++) {
            sum += B2(qy, j) * cache.A2(0, i, qz, qy);
            sum += G2(qy, j) * cache.A2(1, i, qz, qy);
          }
          cache.A1(j, i, qz) = sum;
        }
      }
    }

    for (int k = 0; k < p; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p + 1; i++) {
          double sum = 0.0;
          for (int qz = 0; qz < q; qz++) {
            sum += B1(qz, k) * cache.A1(j, i, qz);
          }
          element_residual.z(k, j, i) += sum;
        }
      }
    }

  }

  template < int q >
  static void integrate(const cpu_batched_values_type<q> & sources,
                        const cpu_batched_derivatives_type<q> & fluxes,
                        const tensor < double, q, q, q, dim, dim > & jacobians,
                        const tensor < double, q, q, q > & jacobians_det,
                        const TensorProductQuadratureRule<q> & rule, 
                        dof_type & element_residual) {

    auto xi = GaussLegendreNodes<q>();

    cache_type<q> cache{};
    tensor<double, q, p > B1;
    tensor<double, q, p+1 > B2;
    tensor<double, q, p+1 > G2;
    for (int i = 0; i < q; i++) {
      B1[i] = GaussLegendreInterpolation<p>(xi[i]);
      B2[i] = GaussLobattoInterpolation<p+1>(xi[i]);
      G2[i] = GaussLobattoInterpolationDerivative<p+1>(xi[i]);
    }

    // transform the source and flux terms from values on the physical element, 
    // to values on the parent element. Also, the source/flux values are scaled
    // according to the weight of their quadrature point, so that 
    for (int qz = 0; qz < q; qz++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          auto detJ = jacobians_det(qz, qy, qx);
          auto J = jacobians(qz, qy, qx);
          sources(qz, qy, qx) = linear_solve(J, sources(qz, qy, qx));
          fluxes(qz, qy, qx) = dot(fluxes(qz, qy, qx), J) / detJ;
        }
      }
    }

    /////////////////////////////////
    ////////// X-component //////////
    /////////////////////////////////
    for (int k = 0; k < p + 1; k++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          double sum[2]{};
          for (int qz = 0; qz < q; qz++) {
            sum[0] += B2(qz, k) * sources(qz, qy, qx, 0) + G2(qz, k) * fluxes(qz, qy, qx, 1);
            sum[1] -= B2(qz, k) * fluxes(qz, qy, qx, 2);
          }
          cache.A2(0, k, qy, qx) = sum[0];
          cache.A2(1, k, qy, qx) = sum[1];
        }
      }
    }

    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int qx = 0; qx < q; qx++) {
          double sum = 0.0;
          for (int qy = 0; qy < q; qy++) {
            sum += B2(qy, j) * cache.A2(0, k, qy, qx);
            sum += G2(qy, j) * cache.A2(1, k, qy, qx);
          }
          cache.A1(k, j, qx) = sum;
        }
      }
    }

    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p; i++) {
          double sum = 0.0;
          for (int qx = 0; qx < q; qx++) {
            sum += B1(qx, i) * cache.A1(k, j, qx);
          }
          element_residual.x(k, j, i) = sum;
        }
      }
    }

    /////////////////////////////////
    ////////// Y-component //////////
    /////////////////////////////////
    for (int k = 0; k < p + 1; k++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          double sum[2]{};
          for (int qz = 0; qz < q; qz++) {
            sum[0] += B2(qz, k) * sources(1, qz, qy, qx) - G2(qz, k) * fluxes(0, qz, qy, qx);
            sum[1] += B2(qz, k) * fluxes(2, qz, qy, qx);
          }
          cache.A2(0, k, qy, qx) = sum[0];
          cache.A2(1, k, qy, qx) = sum[1];
        }
      }
    }

    for (int k = 0; k < p + 1; k++) {
      for (int i = 0; i < p + 1; i++) {
        for (int qy = 0; qy < q; qy++) {
          double sum = 0.0;
          for (int qx = 0; qx < q; qx++) {
            sum += B2(qx, i) * cache.A2(0, k, qy, qx);
            sum += G2(qx, i) * cache.A2(1, k, qy, qx);
          }
          cache.A1(k, i, qy) = sum;
        }
      }
    }

    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p; j++) {
        for (int i = 0; i < p + 1; i++) {
          double sum = 0.0;
          for (int qy = 0; qy < q; qy++) {
            sum += B1(qy, j) * cache.A1(k, i, qy);
          }
          element_residual.y(k, j, i) = sum;
        }
      }
    }

    /////////////////////////////////
    ////////// Z-component //////////
    /////////////////////////////////
    for (int i = 0; i < p + 1; i++) {
      for (int qz = 0; qz < q; qz++) {
        for (int qy = 0; qy < q; qy++) {
          double sum[2]{};
          for (int qx = 0; qx < q; qx++) {
            sum[0] += B2(qx, i) * sources(2, qz, qy, qx) - G2(qx, i) * fluxes(1, qz, qy, qx);
            sum[1] += B2(qx, i) * fluxes(0, qz, qy, qx);
          }
          cache.A2(0, i, qz, qy) = sum[0];
          cache.A2(1, i, qz, qy) = sum[1];
        }
      }
    }

    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p + 1; i++) {
        for (int qz = 0; qz < q; qz++) {
          double sum = 0.0;
          for (int qy = 0; qy < q; qy++) {
            sum += B2(qy, j) * cache.A2(0, i, qz, qy);
            sum += G2(qy, j) * cache.A2(1, i, qz, qy);
          }
          cache.A1(j, i, qz) = sum;
        }
      }
    }

    for (int k = 0; k < p; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p + 1; i++) {
          double sum = 0.0;
          for (int qz = 0; qz < q; qz++) {
            sum += B1(qz, k) * cache.A1(j, i, qz);
          }
          element_residual.z(k, j, i) = sum;
        }
      }
    }

  }

};
/// @endcond

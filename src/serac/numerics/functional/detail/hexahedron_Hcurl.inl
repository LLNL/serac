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

  using residual_type =
      typename std::conditional<components == 1, tensor<double, ndof>, tensor<double, ndof, components>>::type;

  struct mfem_dof_layout {
    tensor< double, p + 1, p + 1, p     > x;
    tensor< double, p + 1, p    , p + 1 > y;
    tensor< double, p    , p + 1, p + 1 > z;
  };

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
  static void interpolation(const mfem_dof_layout & element_values, const TensorProductQuadratureRule<q> &, 
                            tensor<double, p + 1, p + 1, q> & A1, tensor<double, 2, p + 1, q, q> & A2,
                            tensor< double, 3, q, q, q > & value_q, tensor< double, 3, q, q, q > & curl_q) {

    auto xi = GaussLegendreNodes<q>();

    tensor<double, q, p > B1;
    tensor<double, q, p+1 > B2;
    tensor<double, q, p+1 > G2;
    for (int i = 0; i < q; i++) {
      B1[i] = GaussLegendreInterpolation<p>(xi[i]);
      B2[i] = GaussLobattoInterpolation<p+1>(xi[i]);
      G2[i] = GaussLobattoInterpolationDerivative<p+1>(xi[i]);
    }

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
          A1(k, j, qx) = sum;
        }
      }
    }

    for (int k = 0; k < p + 1; k++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          double sum[2]{};
          for (int j = 0; j < (p + 1); j++) {
            sum[0] += B2(qy, j) * A1(k, j, qx);
            sum[1] += G2(qy, j) * A1(k, j, qx);
          }
          A2(0, k, qy, qx) = sum[0];
          A2(1, k, qy, qx) = sum[1];
        }
      }
    }

    for (int qz = 0; qz < q; qz++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          double sum[3]{};
          for (int k = 0; k < (p + 1); k++) {
            sum[0] += B2(qz, k) * A2(0, k, qy, qx);
            sum[1] += G2(qz, k) * A2(0, k, qy, qx);
            sum[2] += B2(qz, k) * A2(1, k, qy, qx);
          }
          value_q(0, qz, qy, qx) += sum[0];
          curl_q(1, qz, qy, qx)  += sum[1];
          curl_q(2, qz, qy, qx)  -= sum[2];
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
          A1(k, i, qy) = sum;
        }
      }
    }

    for (int k = 0; k < p + 1; k++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          double sum[2]{};
          for (int i = 0; i < (p + 1); i++) {
            sum[0] += B2(qx, i) * A1(k, i, qy);
            sum[1] += G2(qx, i) * A1(k, i, qy);
          }
          A2(0, k, qy, qx) = sum[0];
          A2(1, k, qy, qx) = sum[1];
        }
      }
    }

    for (int qz = 0; qz < q; qz++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          double sum[3]{};
          for (int k = 0; k < (p + 1); k++) {
            sum[0] += B2(qz, k) * A2(0, k, qy, qx);
            sum[1] += G2(qz, k) * A2(0, k, qy, qx);
            sum[2] += B2(qz, k) * A2(1, k, qy, qx);
          }
          value_q(1, qz, qy, qx) += sum[0];
          curl_q(2, qz, qy, qx)  += sum[2];
          curl_q(0, qz, qy, qx)  -= sum[1];
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
          A1(j, i, qz) = sum;
        }
      }
    }

    for (int j = 0; j < p + 1; j++) {
      for (int qz = 0; qz < q; qz++) {
        for (int qx = 0; qx < q; qx++) {
          double sum[2]{};
          for (int i = 0; i < (p + 1); i++) {
            sum[0] += B2(qx, i) * A1(j, i, qz);
            sum[1] += G2(qx, i) * A1(j, i, qz);
          }
          A2(0, j, qz, qx) = sum[0];
          A2(1, j, qz, qx) = sum[1];
        }
      }
    }

    for (int qz = 0; qz < q; qz++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          double sum[3]{};
          for (int j = 0; j < (p + 1); j++) {
            sum[0] += B2(qy, j) * A2(0, j, qz, qx);
            sum[1] += G2(qy, j) * A2(0, j, qz, qx);
            sum[2] += B2(qy, j) * A2(1, j, qz, qx);
          }
          value_q(2, qz, qy, qx) += sum[0];
          curl_q(0, qz, qy, qx)  += sum[1];
          curl_q(1, qz, qy, qx)  -= sum[2];
        }
      }
    }

  }

  template < int q >
  static void extrapolation(const tensor< double, 3, q, q, q> & source,
                            const tensor< double, 3, q, q, q> & flux,
                            const TensorProductQuadratureRule<q> &, 
                            tensor< double, 2, p + 1, q, q > & A1, 
                            tensor< double, p + 1, p + 1, q > & A2, 
                            mfem_dof_layout & element_residual) {

    auto xi = GaussLegendreNodes<q>();

    tensor<double, q, p > B1;
    tensor<double, q, p+1 > B2;
    tensor<double, q, p+1 > G2;
    for (int i = 0; i < q; i++) {
      B1[i] = GaussLegendreInterpolation<p>(xi[i]);
      B2[i] = GaussLobattoInterpolation<p+1>(xi[i]);
      G2[i] = GaussLobattoInterpolationDerivative<p+1>(xi[i]);
    }

    /////////////////////////////////
    ////////// X-component //////////
    /////////////////////////////////
    for (int k = 0; k < p + 1; k++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          double sum[2]{};
          for (int qz = 0; qz < q; qz++) {
            sum[0] += B2(qz, k) * source(0, qz, qy, qx) + G2(qz, k) * flux(1, qz, qy, qx);
            sum[1] -= B2(qz, k) * flux(2, qz, qy, qx);
          }
          A1(0, k, qy, qx) = sum[0];
          A1(1, k, qy, qx) = sum[1];
        }
      }
    }

    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int qx = 0; qx < q; qx++) {
          double sum = 0.0;
          for (int qy = 0; qy < q; qy++) {
            sum += B2(qy, j) * A1(0, k, qy, qx);
            sum += G2(qy, j) * A1(1, k, qy, qx);
          }
          A2(k, j, qx) = sum;
        }
      }
    }

    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p; i++) {
          double sum = 0.0;
          for (int qx = 0; qx < q; qx++) {
            sum += B1(qx, i) * A2(k, j, qx);
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
            sum[0] += B2(qz, k) * source(1, qz, qy, qx) - G2(qz, k) * flux(0, qz, qy, qx);
            sum[1] += B2(qz, k) * flux(2, qz, qy, qx);
          }
          A1(0, k, qy, qx) = sum[0];
          A1(1, k, qy, qx) = sum[1];
        }
      }
    }

    for (int k = 0; k < p + 1; k++) {
      for (int i = 0; i < p + 1; i++) {
        for (int qy = 0; qy < q; qy++) {
          double sum = 0.0;
          for (int qx = 0; qx < q; qx++) {
            sum += B2(qx, i) * A1(0, k, qy, qx);
            sum += G2(qx, i) * A1(1, k, qy, qx);
          }
          A2(k, i, qy) = sum;
        }
      }
    }

    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p; j++) {
        for (int i = 0; i < p + 1; i++) {
          double sum = 0.0;
          for (int qy = 0; qy < q; qy++) {
            sum += B1(qy, j) * A2(k, i, qy);
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
            sum[0] += B2(qx, i) * source(2, qz, qy, qx) - G2(qx, i) * flux(1, qz, qy, qx);
            sum[1] += B2(qx, i) * flux(0, qz, qy, qx);
          }
          A1(0, i, qz, qy) = sum[0];
          A1(1, i, qz, qy) = sum[1];
        }
      }
    }

    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p + 1; i++) {
        for (int qz = 0; qz < q; qz++) {
          double sum = 0.0;
          for (int qy = 0; qy < q; qy++) {
            sum += B2(qy, j) * A1(0, i, qz, qy);
            sum += G2(qy, j) * A1(1, i, qz, qy);
          }
          A2(j, i, qz) = sum;
        }
      }
    }

    for (int k = 0; k < p; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p + 1; i++) {
          double sum = 0.0;
          for (int qz = 0; qz < q; qz++) {
            sum += B1(qz, k) * A2(j, i, qz);
          }
          element_residual.z(k, j, i) = sum;
        }
      }
    }

  }

};
/// @endcond

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
  static constexpr int  ndof       = 2 * p * (p + 1);
  static constexpr int  components = 1;

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

  using residual_type =
      typename std::conditional<components == 1, tensor<double, ndof>, tensor<double, ndof, components> >::type;

  struct mfem_dof_layout {
    tensor< double, p + 1, p     > x;
    tensor< double, p    , p + 1 > y;
  };

  /*

    interpolation nodes/directions and their associated numbering:

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

  template < int q >
  static void interpolation(const mfem_dof_layout & element_values, const TensorProductQuadratureRule<q> &, 
                            tensor<double, p + 1, q> & A1,
                            tensor< double, 2, q, q > & value_q, tensor< double, q, q > & curl_q) {

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
    for (int j = 0; j < p + 1; j++) {
      for (int qx = 0; qx < q; qx++) {
        double sum = 0.0;
        for (int i = 0; i < p; i++) {
          sum += B1(qx, i) * element_values.x(j, i);
        }
        A1(j, qx) = sum;
      }
    }
    
    for (int qy = 0; qy < q; qy++) {
      for (int qx = 0; qx < q; qx++) {
        double sum[2]{};
        for (int j = 0; j < (p + 1); j++) {
          sum[0] += B2(qy, j) * A1(j, qx);
          sum[1] += G2(qy, j) * A1(j, qx);
        }
        value_q(0, qy, qx) =  sum[0];
        curl_q(qy, qx)     = -sum[1];
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
        A1(i, qy) = sum;
      }
    }
    
    for (int qy = 0; qy < q; qy++) {
      for (int qx = 0; qx < q; qx++) {
        double sum[3]{};
        for (int i = 0; i < (p + 1); i++) {
          sum[0] += B2(qx, i) * A1(i, qy);
          sum[1] += G2(qx, i) * A1(i, qy);
        }
        value_q(1, qy, qx) = sum[0];
        curl_q(qy, qx)     += sum[1];
      }
    }

  }

  template < int q >
  static void extrapolation(const tensor< double, 2, q, q> & source,
                            const tensor< double, q, q> & flux,
                            const TensorProductQuadratureRule<q> &, 
                            tensor< double, p + 1, q > & A1, 
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
    for (int j = 0; j < (p + 1); j++) {
      for (int qx = 0; qx < q; qx++) {
        double sum = 0.0;
        for (int qy = 0; qy < q; qy++) {
          sum += B2(qy, j) * source(0, qy, qx) - G2(qy, j) * flux(qy, qx);
        }
        A1(j, qx) = sum;
      }
    }
    
    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p; i++) {
        double sum = 0.0;
        for (int qx = 0; qx < q; qx++) {
          sum += B1(qx, i) * A1(j, qx);
        }
        element_residual.x(j, i) = sum;
      }
    }

    /////////////////////////////////
    ////////// Y-component //////////
    /////////////////////////////////
    for (int i = 0; i < (p + 1); i++) {
      for (int qy = 0; qy < q; qy++) {
        double sum = 0.0;
        for (int qx = 0; qx < q; qx++) {
          sum += B2(qx, i) * source(1, qy, qx) + G2(qx, i) * flux(qy, qx);
        }
        A1(i, qy) = sum;
      }
    }
    
    for (int j = 0; j < p; j++) {
      for (int i = 0; i < p + 1; i++) {
        double sum = 0.0;
        for (int qy = 0; qy < q; qy++) {
          sum += B1(qy, j) * A1(i, qy);
        }
        element_residual.y(j, i) = sum;
      }
    }

  }



};
/// @endcond

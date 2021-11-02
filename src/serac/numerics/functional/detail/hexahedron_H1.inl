// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
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
  static constexpr int  ndof       = (p + 1) * (p + 1) * (p + 1);

  using residual_type =
      typename std::conditional<components == 1, tensor<double, ndof>, tensor<double, ndof, components> >::type;

  static constexpr tensor<double, ndof> shape_functions(tensor<double, dim> xi)
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

  static constexpr tensor<double, ndof, dim> shape_function_gradients(tensor<double, dim> xi)
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
};
/// @endcond

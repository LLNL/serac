// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file quadrilateral_L2.inl
 *
 * @brief Specialization of finite_element for L2 on quadrilateral geometry
 */

// specialization of finite_element for L2 on quadrilateral geometry
//
// this specialization defines shape functions that interpolate 
// at Gauss-Lobatto nodes for the appropriate polynomial order
// 
// note 1: mfem assumes the parent element domain is [0,1]x[0,1]
// note 2: mfem doesn't define "diffusion" integrators on L2, so we don't define the shape function gradients
template <int p>
struct finite_element<Geometry::Quadrilateral, L2<p> > {
  static constexpr auto geometry   = Geometry::Quadrilateral;
  static constexpr auto family     = Family::L2;
  static constexpr int  dim        = 2;
  static constexpr int  ndof       = (p + 1) * (p + 1);

  static constexpr tensor<double, ndof> shape_functions(tensor<double, dim> xi)
  {
    auto N_xi  = GaussLobattoInterpolation01<p + 1>(xi[0]);
    auto N_eta = GaussLobattoInterpolation01<p + 1>(xi[1]);

    int count = 0;

    tensor<double, ndof> N{};
    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p + 1; i++) {
        N[count++] = N_xi[i] * N_eta[j];
      }
    }
    return N;
  }

};

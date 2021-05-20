// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file segment_h1.inl
 *
 * @brief Specialization of finite_element for H1 on segment geometry
 */

// this specialization defines shape functions (and their gradients) that
// interpolate at Gauss-Lobatto nodes for the appropriate polynomial order
// 
// note: mfem assumes the parent element domain is [0,1]
template <int p, int c>
struct finite_element<Geometry::Segment, H1< p, c > > {

  static constexpr auto geometry = Geometry::Segment;
  static constexpr auto family = Family::H1;
  static constexpr int components = c;
  static constexpr int dim = 1;
  static constexpr int ndof = (p + 1);

  static constexpr tensor<double, ndof> shape_functions(double xi) {
    return GaussLobattoInterpolation01<ndof>(xi);
  }

  static constexpr tensor<double, ndof> shape_function_gradients(double xi) {
    return GaussLobattoInterpolationDerivative01<ndof>(xi);
  }
};

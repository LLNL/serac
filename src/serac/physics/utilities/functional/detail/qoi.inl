// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file qoi.inl
 *
 * @brief Specialization of finite_element for expressing quantities of interest on any geometry
 */
template < Geometry g >
struct finite_element< g, QOI > {
  static constexpr auto geometry   = g;
  static constexpr auto family     = Family::QOI;
  static constexpr int  components = 1;
  static constexpr int  dim        = 1;
  static constexpr int  ndof       = 1;

  using residual_type = double;

  static constexpr double shape_functions(double /* xi */) { return 1.0; }
}; 

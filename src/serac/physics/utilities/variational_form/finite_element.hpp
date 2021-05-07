
// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file finite_element.hpp
 *
 * @brief This file contains helper traits and enumerations for classifying
 * finite elements
 */
#pragma once

#include "tensor.hpp"
#include "polynomials.hpp"

namespace serac {

/**
 * @brief Element geometries
 */
enum class Geometry
{
  Point,
  Segment,
  Triangle,
  Quadrilateral,
  Tetrahedron,
  Hexahedron
};

/**
 * @brief Compile-time alias for a dimension
 */
template <int d>
struct Dimension {
  /**
   * @brief Returns the dimension
   */
  constexpr operator int() { return d; }
};

/**
 * @brief Returns the dimension of an element geometry
 * @param[in] g The @p Geometry to retrieve the dimension of
 */
constexpr int dimension_of(Geometry g)
{
  if (g == Geometry::Segment) {
    return 1;
  }

  if (g == Geometry::Triangle || g == Geometry::Quadrilateral) {
    return 2;
  }

  if (g == Geometry::Tetrahedron || g == Geometry::Hexahedron) {
    return 3;
  }

  return -1;
}

/**
 * @brief H1 elements of order @p p
 * @tparam p The order of the elements
 * @tparam c The vector dimension
 */
template <int p, int c = 1>
struct H1 {
  static constexpr int order      = p;
  static constexpr int components = c;
};

/**
 * @brief H(curl) elements of order @p p
 * @tparam p The order of the elements
 * @tparam c The vector dimension
 */
template <int p, int c = 1>
struct Hcurl {
  static constexpr int order      = p;
  static constexpr int components = c;
};

/**
 * @brief Discontinuous elements of order @p p
 * @tparam p The order of the elements
 * @tparam c The vector dimension
 */
template <int p, int c = 1>
struct L2 {
  static constexpr int order      = p;
  static constexpr int components = c;
};

/**
 * @brief Element conformity
 */
enum class Family
{
  H1,
  HCURL,
  HDIV,
  L2
};

/**
 * @brief FIXME: This doesn't appear to be used anywhere
 */
enum class Evaluation
{
  Interpolate,
  Divergence,
  Gradient,
  Curl
};

/**
 * @brief Template prototype for finite element implementations
 * @tparam g The geometry of the element
 * @tparam family The continuity of the element
 */
template <Geometry g, typename family>
struct finite_element;

/**
 * @brief Type trait for identifying finite element types
 */
template <typename T>
struct is_finite_element {
  static constexpr bool value = false;
};
/// @overload
template <Geometry g, int p, int c>
struct is_finite_element<finite_element<g, H1<p, c> > > {
  static constexpr bool value = true;
};
/// @overload
template <Geometry g, int p>
struct is_finite_element<finite_element<g, Hcurl<p> > > {
  static constexpr bool value = true;
};

#include "detail/segment_h1.inl"
//#include "detail/segment_hcurl.inl"

//#include "detail/triangle_h1.inl"
//#include "detail/triangle_hcurl.inl"

#include "detail/quadrilateral_h1.inl"
#include "detail/quadrilateral_hcurl.inl"
#include "detail/quadrilateral_L2.inl"

//#include "detail/tetrahedron_h1.inl"
//#include "detail/tetrahedron_hcurl.inl"

#include "detail/hexahedron_h1.inl"
#include "detail/hexahedron_hcurl.inl"

}  // namespace serac

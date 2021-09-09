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
 * @brief H1 elements of order @p p
 * @tparam p The order of the elements
 * @tparam c The vector dimension
 */
template <int p, int c = 1>
struct H1 {
  static constexpr int    order      = p;  ///< the polynomial order of the elements
  static constexpr int    components = c;  ///< the number of components at each node
  static constexpr Family family     = Family::H1;
};

/**
 * @brief H(curl) elements of order @p p
 * @tparam p The order of the elements
 * @tparam c The vector dimension
 */
template <int p, int c = 1>
struct Hcurl {
  static constexpr int    order      = p;  ///< the polynomial order of the elements
  static constexpr int    components = c;  ///< the number of components at each node
  static constexpr Family family     = Family::HCURL;
};

/**
 * @brief Discontinuous elements of order @p p
 * @tparam p The order of the elements
 * @tparam c The vector dimension
 */
template <int p, int c = 1>
struct L2 {
  static constexpr int    order      = p;  ///< the polynomial order of the elements
  static constexpr int    components = c;  ///< the number of components at each node
  static constexpr Family family     = Family::L2;
};

/**
 * @brief Template prototype for finite element implementations
 * @tparam g The geometry of the element
 * @tparam family The continuity of the element
 * the implementations of the different finite element specializations
 * are in .inl files in the detail/ directory.
 *
 * In each of these files, the finite_element specialization
 * should implement the following concept:
 *
 * struct finite_element< some_geometry, some_space > > {
 *   static constexpr Geometry geometry = ...; ///< one of Triangle, Quadrilateral, etc
 *   static constexpr Family family     = ...; ///< one of H1, HCURL, HDIV, etc
 *   static constexpr int  components   = ...; ///< how many components per node
 *   static constexpr int  dim          = ...; ///< number of parent element coordinates
 *   static constexpr int  ndof         = ...; ///< how many degrees of freedom for an element with 1 component per node
 *
 *   /// implement the way this element type interpolates the solution on the interior of the element
 *   static constexpr auto shape_functions(tensor<double, dim> xi) { ... }
 *
 *   /// implement the derivatives of this element's shape functions w.r.t. parent element coordinates
 *   static constexpr auto shape_function_derivatives(tensor<double, dim> xi) { ... }
 * };
 *
 */
template <Geometry g, typename family>
struct finite_element;

/**
 * @brief Type trait for identifying finite element types
 */
template <typename T>
struct is_finite_element {
  static constexpr bool value = false;  ///< whether or not type T is a finite_element
};
/// @overload
template <Geometry g, int p, int c>
struct is_finite_element<finite_element<g, H1<p, c> > > {
  static constexpr bool value = true;  ///< whether or not type T is a finite_element
};
/// @overload
template <Geometry g, int p>
struct is_finite_element<finite_element<g, Hcurl<p> > > {
  static constexpr bool value = true;  ///< whether or not type T is a finite_element
};

/**
 * @brief a class that helps to extract the test space from a function signature template parameter
 * @tparam space The function signature itself
 */
template <typename spaces>
struct get_test_space;  // undefined

/**
 * @brief a class that helps to extract the test space from a function signature template parameter
 * @tparam space The function signature itself
 */
template <typename test_space, typename trial_space>
struct get_test_space<test_space(trial_space)> {
  using type = test_space;  ///< the test space
};

/**
 * @brief a class that helps to extract the trial space from a function signature template parameter
 * @tparam space The function signature itself
 */
template <typename spaces>
struct get_trial_space;  // undefined

/**
 * @brief a class that helps to extract the trial space from a function signature template parameter
 * @tparam space The function signature itself
 */
template <typename test_space, typename trial_space>
struct get_trial_space<test_space(trial_space)> {
  using type = trial_space;  ///< the trial space
};

/**
 * @brief a type function that extracts the test space from a function signature template parameter
 * @tparam space The function signature itself
 */
template <typename spaces>
using test_space_t = typename get_test_space<spaces>::type;

/**
 * @brief a type function that extracts the trial space from a function signature template parameter
 * @tparam space The function signature itself
 */
template <typename spaces>
using trial_space_t = typename get_trial_space<spaces>::type;

#include "detail/segment_H1.inl"
#include "detail/segment_Hcurl.inl"
#include "detail/segment_L2.inl"

#include "detail/quadrilateral_H1.inl"
#include "detail/quadrilateral_Hcurl.inl"
#include "detail/quadrilateral_L2.inl"

#include "detail/hexahedron_H1.inl"
#include "detail/hexahedron_Hcurl.inl"
#include "detail/hexahedron_L2.inl"

}  // namespace serac

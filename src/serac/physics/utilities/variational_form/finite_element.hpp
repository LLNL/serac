#pragma once

#include "tensor.hpp"
#include "polynomials.hpp"

enum class Geometry {Segment, Triangle, Quadrilateral, Tetrahedron, Hexahedron};

enum class Family {H1, HCURL, HDIV};

enum class PolynomialDegree {Constant = 0, Linear = 1, Quadratic = 2, Cubic = 3};

enum class Evaluation {Interpolate, Divergence, Gradient, Curl};

template < Geometry g, Family f, PolynomialDegree p, int components = 1 >
struct finite_element;

template < typename T >
struct is_finite_element {
  static constexpr bool value = false;
};

template < Geometry g, Family f, PolynomialDegree p, int c >
struct is_finite_element< finite_element< g, f, p, c > >{
  static constexpr bool value = true;
};

template < typename T >
struct quadrature_data {
  using type = T;
};

template < typename T >
struct is_quadrature_data {
  static constexpr bool value = false;
};

template < typename T >
struct is_quadrature_data< quadrature_data < T > >{
  static constexpr bool value = true;
};

#include "detail/segment_h1.inl"
//#include "detail/segment_hcurl.inl"

//#include "detail/triangle_h1.inl"
//#include "detail/triangle_hcurl.inl"

#include "detail/quadrilateral_h1.inl"
#include "detail/quadrilateral_hcurl.inl"

//#include "detail/tetrahedron_h1.inl"
//#include "detail/tetrahedron_hcurl.inl"

#include "detail/hexahedron_h1.inl"
//#include "detail/hexahedron_hcurl.inl"
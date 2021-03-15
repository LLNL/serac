#pragma once

#include "tensor.hpp"
#include "polynomials.hpp"

enum class Geometry {Segment, Triangle, Quadrilateral, Tetrahedron, Hexahedron};

constexpr int dimension(::Geometry g) {
  if (g == ::Geometry::Segment) {
    return 1;
  }

  if (g == ::Geometry::Triangle || g == ::Geometry::Quadrilateral) {
    return 2;
  }

  if (g == ::Geometry::Tetrahedron || g == ::Geometry::Hexahedron) {
    return 3;
  }

  return -1;
}

template < int p, int c = 1 >
struct H1{
  static constexpr int order = p;
  static constexpr int components = c;
};

template < int p, int c = 1 >
struct Hcurl{
  static constexpr int order = p;
  static constexpr int components = c;
};

template < int p >
struct L2{
  static constexpr int order = p;
};

enum class Family {H1, HCURL, HDIV, L2};

enum class Evaluation {Interpolate, Divergence, Gradient, Curl};

template < ::Geometry g, typename family >
struct finite_element;

template < typename T >
struct is_finite_element {
  static constexpr bool value = false;
};

template < ::Geometry g, int p, int c >
struct is_finite_element< finite_element< g, H1<p, c> > >{
  static constexpr bool value = true;
};

template < ::Geometry g, int p >
struct is_finite_element< finite_element< g, Hcurl<p> > >{
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
#include "detail/quadrilateral_L2.inl"

//#include "detail/tetrahedron_h1.inl"
//#include "detail/tetrahedron_hcurl.inl"

#include "detail/hexahedron_h1.inl"
//#include "detail/hexahedron_hcurl.inl"
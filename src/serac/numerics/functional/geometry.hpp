#pragma once

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
 * @brief return the number of quadrature points in a Gauss-Legendre rule
 * with parameter "q"
 *
 * @tparam g the element geometry
 * @tparam q the number of quadrature points per dimension
 */
constexpr int num_quadrature_points(Geometry g, int q)
{
  if (g == Geometry::Segment) {
    return q;
  }
  if (g == Geometry::Triangle) {
    return (q * (q + 1)) / 2;
  }
  if (g == Geometry::Quadrilateral) {
    return q * q;
  }
  if (g == Geometry::Tetrahedron) {
    return (q * (q + 1) * (q + 2)) / 6;
  }
  if (g == Geometry::Hexahedron) {
    return q * q * q;
  }
  return -1;
}

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

}

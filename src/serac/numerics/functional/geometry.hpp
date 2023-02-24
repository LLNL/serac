#pragma once

namespace serac {

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
constexpr int num_quadrature_points(mfem::Geometry::Type g, int q)
{
  if (g == mfem::Geometry::SEGMENT) {
    return q;
  }
  if (g == mfem::Geometry::TRIANGLE) {
    return (q * (q + 1)) / 2;
  }
  if (g == mfem::Geometry::SQUARE) {
    return q * q;
  }
  if (g == mfem::Geometry::TETRAHEDRON) {
    return (q * (q + 1) * (q + 2)) / 6;
  }
  if (g == mfem::Geometry::CUBE) {
    return q * q * q;
  }
  return -1;
}

/**
 * @brief Returns the dimension of an element geometry
 * @param[in] g The @p Geometry to retrieve the dimension of
 */
constexpr int dimension_of(mfem::Geometry::Type g)
{
  if (g == mfem::Geometry::SEGMENT) {
    return 1;
  }

  if (g == mfem::Geometry::TRIANGLE || g == mfem::Geometry::SQUARE) {
    return 2;
  }

  if (g == mfem::Geometry::TETRAHEDRON || g == mfem::Geometry::CUBE) {
    return 3;
  }

  return -1;
}

}  // namespace serac

#pragma once

#include "serac/numerics/functional/element_restriction.hpp"  // for FaceType
#include "serac/numerics/functional/finite_element.hpp"       // for Geometry
#include "serac/numerics/functional/domain.hpp"

#include "mfem.hpp"

namespace serac {

/**
 * @brief a class that computes and stores positions and jacobians at each quadrature point
 * @note analogous to mfem::GeometricFactors, except that it implements the position/jacobian
 *       calculations on boundary elements and on simplex elements
 */
struct GeometricFactors {
  /// @brief default ctor, leaving this object uninitialized
  GeometricFactors(){};

  /**
   * @brief calculate positions and jacobians for quadrature points belonging to
   * elements with the specified geometry, belonging to the provided mesh.
   *
   * @param domain the domain of integration
   * @param q a parameter controlling the number of quadrature points per element
   * @param elem_geom which kind of element geometry to select
   */
  GeometricFactors(const Domain& domain, int q, mfem::Geometry::Type elem_geom);

  /**
   * @brief calculate positions and jacobians for quadrature points belonging to
   * boundary elements with the specified geometry, belonging to the provided mesh.
   *
   * @param domain the domain of integration
   * @param q a parameter controlling the number of quadrature points per element
   * @param elem_geom which kind of element geometry to select
   * @param type whether or not the faces are on the boundary (supported) or interior (unsupported)
   */
  GeometricFactors(const Domain& domain, int q, mfem::Geometry::Type elem_geom, FaceType type);

  // descriptions copied from mfem

  /// Mapped (physical) coordinates of all quadrature points.
  /** This array uses a column-major layout with dimensions (NQ x SDIM x NE)
      where
      - NQ = number of quadrature points per element,
      - SDIM = space dimension of the mesh = mesh.SpaceDimension(), and
      - NE = number of elements in the mesh. */
  mfem::Vector X;

  /// Jacobians of the element transformations at all quadrature points.
  /** This array uses a column-major layout with dimensions (NQ x SDIM x DIM x
      NE) where
      - NQ = number of quadrature points per element,
      - SDIM = space dimension of the mesh = mesh.SpaceDimension(),
      - DIM = dimension of the mesh = mesh.Dimension(), and
      - NE = number of elements in the mesh. */
  mfem::Vector J;

  /// @brief list of element indices that are part of the associated domain
  std::vector<int> elements;

  /// the number of elements in the domain
  std::size_t num_elements;
};

}  // namespace serac

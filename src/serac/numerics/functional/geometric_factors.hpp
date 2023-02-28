#pragma once

#include "serac/numerics/functional/element_restriction.hpp"  // for FaceType
#include "serac/numerics/functional/finite_element.hpp"       // for Geometry

#include "mfem.hpp"

namespace serac {

struct GeometricFactors {

  GeometricFactors(){};
  GeometricFactors(const mfem::Mesh* mesh, int q, mfem::Geometry::Type elem_geom);
  GeometricFactors(const mfem::Mesh* mesh, int q, mfem::Geometry::Type elem_geom, FaceType type);

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

  std::size_t num_elements;
};

}  // namespace serac

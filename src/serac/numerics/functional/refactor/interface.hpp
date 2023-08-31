#pragma once

#include <tuple>

#include "axom/core.hpp"
#include "serac/numerics/functional/finite_element.hpp"
#include "serac/physics/state/finite_element_state.hpp"
#include "serac/physics/state/finite_element_dual.hpp"

namespace serac {

static constexpr std::array supported_geometries = {mfem::Geometry::SEGMENT,
                                                    mfem::Geometry::TRIANGLE,
                                                    mfem::Geometry::SQUARE,
                                                    mfem::Geometry::TETRAHEDRON,
                                                    mfem::Geometry::CUBE};

using geom_array = std::array<uint32_t, mfem::Geometry::NUM_GEOMETRIES>;

std::tuple<geom_array, uint32_t> quadrature_point_offsets(const mfem::Mesh & mesh, int q);

/**
 * @brief evaluate the solution field `u` at each
 *        quadrature point in the mesh,
 *        where each element type is using a
 *        Gauss-Legendre rule of order `q`
 *
 * @param u_q output, the quadrature point values
 * @param u input, the nodal values
 * @param q parameter controlling the number of quadrature points per element
 */
void interpolate(axom::Array<double, 2>& u_q, const FiniteElementState& u, int q);

/**
 * @brief evaluate the gradient of the solution field `u` at each
 *        quadrature point in the mesh,
 *        where each element type is using a
 *        Gauss-Legendre rule of order `q`
 *
 * @param du_dX_q output, the quadrature point gradients w.r.t. reference coordinates
 * @param u input, the nodal values
 * @param q parameter controlling the number of quadrature points per element
 */
void gradient(axom::Array<double, 3>& du_dX_q, const FiniteElementState& u, int q);

/**
 * @brief integrate the source and flux terms against test functions
 *        from the space specified by `f`:
 *
 *        f_i := \int s(X) \phi_i(X) + dot(f(X), \nabla \phi_i(X)) dX
 *
 * @param f output, nodal "forces" calculated by integrating the source and flux terms
 * @param source input, "source term" (e.g. body force, heat source) at each quadrature point
 * @param flux input, "flux term" (e.g. PK stress, heat flux) at each quadrature point
 * @param q parameter controlling the number of quadrature points per element
 */
void integrate(FiniteElementDual& f, const axom::Array<double, 2> source, const axom::Array<double, 3> flux, int q);

}  // namespace serac

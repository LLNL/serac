#pragma once

#include "axom/core.hpp"
#include "serac/numerics/functional/finite_element.hpp"
#include "serac/physics/state/finite_element_state.hpp"

namespace serac {

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
void interpolate(axom::Array<double, 3>& u_q, const FiniteElementState& u, int q);

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
void gradient(axom::Array<double, 4>& du_dX_q, const FiniteElementState& u, int q);

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
void integrate(FiniteElementDual& f, const axom::Array<double, 3> source, const axom::Array<double, 4> flux, int q);

}  // namespace serac

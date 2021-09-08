#pragma once

#include "serac/physics/utilities/functional/tensor.hpp"
#include "serac/physics/utilities/functional/quadrature.hpp"
#include "serac/physics/utilities/functional/finite_element.hpp"

template <Geometry g, typename function_space, int Q, int dim>
inline void calculate_boundary_integration_points(const mfem::Vector& x_e, Array2D<tensor<dim + 1> >& x_q,
                                                  Array2D<tensor<dim + 1, dim> >& j_q)
{
  using element = finite_element<g, test>;

  int                   num_boundary_elements = x_q.size();
  static constexpr int  ndof                  = element::ndof;
  static constexpr auto rule                  = GaussQuadratureRule<g, Q>();

  // mfem passes around multidimensional data as flattened 1D arrays,
  // so we have to reinterpret it as a multidimensional array of the appropriate size
  auto X_E = detail::Reshape<trial>(x_e.Read(), dim + 1, ndof, num_elements);

  for (int e = 0; e < num_boundary_elements; e++) {
    // get the DOF values for this particular element
    tensor x_elem = detail::Load<element>(X_e, e);

    // for each quadrature point in the element
    for (int q = 0; q < static_cast<int>(rule.size()); q++) {
      // get the position of this quadrature point in the parent space
      auto xi = rule.points[q];

      // and use it to calculate the required quantities:
      x_q(e, q) = dot(x_elem, element::shape_functions(xi));

      // derivatives of spatial position w.r.t. (xi, eta)
      J_q(e, q) = dot(x_elem, element::shape_function_gradients(xi));
    }
  }
}

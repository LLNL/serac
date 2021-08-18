#pragma once

#include "serac/physics/utilities/functional/tensor.hpp"

namespace serac {

namespace domain_integral {

/**
 * @brief Computes the arguments to be passed into the q-function (shape function evaluations)
 * By default:
 *  H1 family elements will compute {value, gradient}
 *  Hcurl family elements will compute {value, curl}
 *  TODO: Hdiv family elements will compute {value, divergence}
 *  L2 family elements will compute value
 *
 * In the future, the user will be able to override these defaults
 * to omit unused components (e.g. specify that they only need the gradient)
 *
 * @param[in] u The DOF values for the element
 * @param[in] xi The position of the quadrature point in reference space
 * @param[in] J The Jacobian of the element transformation at the quadrature point
 * @tparam element_type The type of the element (used to determine the family)
 */
template <typename element_type, typename T, int dim>
SERAC_HOST_DEVICE auto Preprocess(T u, const tensor<double, dim> xi, const tensor<double, dim, dim> J)
{
  if constexpr (element_type::family == Family::H1 || element_type::family == Family::L2) {
    return serac::tuple{dot(u, element_type::shape_functions(xi)),
                        dot(u, dot(element_type::shape_function_gradients(xi), inv(J)))};
  }

  if constexpr (element_type::family == Family::HCURL) {
    // HCURL shape functions undergo a covariant Piola transformation when going
    // from parent element to physical element
    auto value = dot(u, dot(element_type::shape_functions(xi), inv(J)));
    auto curl  = dot(u, element_type::shape_function_curl(xi) / det(J));
    if constexpr (dim == 3) {
      curl = dot(curl, transpose(J));
    }
    return serac::tuple{value, curl};
  }
}

/**
 * @brief Computes residual contributions from the output of the q-function
 * This involves integrating the q-function output against functions from the
 * test function space.
 *
 * By default:
 *  H1 family elements integrate std::get<0>(f) against the test space shape functions
 *                           and std::get<1>(f) against the test space shape function gradients
 *  Hcurl family elements integrate std::get<0>(f) against the test space shape functions
 *                              and std::get<1>(f) against the curl of the test space shape functions
 * TODO: Hdiv family elements integrate std::get<0>(f) against the test space shape functions
 *                                  and std::get<1>(f) against the divergence of the test space shape functions
 *  L2 family elements integrate f against test space shape functions
 *
 * In the future, the user will be able to override these defaults
 * to omit unused components (e.g. provide only the term to be integrated against test function gradients)
 * @tparam element_type The type of the element (used to determine the family)
 * @tparam T The type of the output from the user-provided q-function
 * @pre T must be a pair type for H1, H(curl) and H(div) family elements
 * @param[in] f The value component output of the user's quadrature function (as opposed to the value/derivative pair)
 * @param[in] xi The position of the quadrature point in reference space
 * @param[in] J The Jacobian of the element transformation at the quadrature point
 */
template <typename element_type, typename T, int dim>
SERAC_HOST_DEVICE auto Postprocess(T f, const tensor<double, dim> xi, const tensor<double, dim, dim> J)
{
  // TODO: Helpful static_assert about f being tuple or tuple-like for H1, hcurl, hdiv
  if constexpr (element_type::family == Family::H1 || element_type::family == Family::L2) {
    auto W     = element_type::shape_functions(xi);
    auto dW_dx = dot(element_type::shape_function_gradients(xi), inv(J));
    return outer(W, serac::get<0>(f)) + dot(dW_dx, serac::get<1>(f));
  }

  if constexpr (element_type::family == Family::HCURL) {
    auto W      = dot(element_type::shape_functions(xi), inv(J));
    auto curl_W = element_type::shape_function_curl(xi) / det(J);
    if constexpr (dim == 3) {
      curl_W = dot(curl_W, transpose(J));
    }
    return (W * serac::get<0>(f) + curl_W * serac::get<1>(f));
  }
}

/**
 * @brief A host/device template used to calculate element residuals at the quadrature point level
 *
 * @tparam test The type of the test function space
 * @tparam trial The type of the trial function space
 * The above spaces can be any combination of {H1, Hcurl, Hdiv (TODO), L2 (TODO)}
 *
 * Template parameters other than the test and trial spaces are used for customization + optimization
 * and are erased through the @p std::function members of @p DomainIntegral
 * @tparam g The shape of the element (only quadrilateral and hexahedron are supported at present)
 * @tparam Q Quadrature parameter describing how many points per dimension
 * @tparam derivatives_type Type representing the derivative of the q-function (see below) w.r.t. its input arguments
 * @tparam lambda The actual quadrature-function (either lambda function or functor object) to
 * be evaluated at each quadrature point.
 * @see https://libceed.readthedocs.io/en/latest/libCEEDapi/#theoretical-framework for additional
 * information on the idea behind a quadrature function and its inputs/outputs
 * @tparam qpt_data_type The type of the data to store for each quadrature point
 *
 * @param[in] e The element number
 * @param[in] q The quadrature point index
 * @param[in] u_elem The element DOF values (primary input)
 * @param[inout] r_elem The element residuals (primary output)
 * @param[out] derivatives_ptr The address at which derivatives of @a lambda with
 * respect to its arguments will be stored
 * @param[in] J The Jacobian of the element transformation at all quadrature points
 * @param[in] X The actual (not reference) coordinates of all quadrature points
 * @see mfem::GeometricFactors
 * @param[in] num_elements The number of elements in the mesh
 * @param[in] qf The actual quadrature function, see @p lambda
 */
template <Geometry g, typename test, typename trial, int Q, typename derivatives_type, typename lambda,
          typename u_elem_type, typename element_residual_type, typename J_type, typename X_type>
SERAC_HOST_DEVICE void eval_quadrature(int e, int q, u_elem_type u_elem, element_residual_type& r_elem,
                                       derivatives_type* derivatives_ptr, J_type J, X_type X, int num_elements,
                                       lambda qf)
{
  using test_element         = finite_element<g, test>;
  using trial_element        = finite_element<g, trial>;
  static constexpr auto rule = GaussQuadratureRule<g, Q>();
  static constexpr int  dim  = dimension_of(g);

  // get the position of this quadrature point in the parent and physical space,
  // and calculate the measure of that point in physical space.

  auto   xi  = rule.points[q];
  auto   dxi = rule.weights[q];
  auto   x_q = make_tensor<dim>([&](int i) { return X(q, i, e); });  // Physical coords of qpt
  auto   J_q = make_tensor<dim, dim>([&](int i, int j) { return J(q, i, j, e); });
  double dx  = det(J_q) * dxi;

  // evaluate the value/derivatives needed for the q-function at this quadrature point
  auto arg = Preprocess<trial_element>(u_elem, xi, J_q);

  // evaluate the user-specified constitutive model
  //
  // note: make_dual(arg) promotes those arguments to dual number types
  // so that qf_output will contain values and derivatives
  auto qf_output = qf(x_q, make_dual(arg));

  // integrate qf_output against test space shape functions / gradients
  // to get element residual contributions
  r_elem += Postprocess<test_element>(get_value(qf_output), xi, J_q) * dx;

  // here, we store the derivative of the q-function w.r.t. its input arguments
  //
  // this will be used by other kernels to evaluate gradients / adjoints / directional derivatives

  // Note: This pattern may result in non-coalesced access depend on how it executed.
  detail::AccessDerivatives(derivatives_ptr, e, q, rule, num_elements) = get_gradient(qf_output);
}

/**
 * @brief This a host/device template used to calculate gradients at a quadrature point
 * kernels associated with finite element calculations
 *
 * @tparam test The type of the test function space
 * @tparam trial The type of the trial function space
 * The above spaces can be any combination of {H1, Hcurl, Hdiv (TODO), L2 (TODO)}
 *
 * Template parameters other than the test and trial spaces are used for customization + optimization
 * and are erased through the @p std::function members of @p DomainIntegral
 * @tparam g The shape of the element (only quadrilateral and hexahedron are supported at present)
 * @tparam Q Quadrature parameter describing how many points per dimension
 * @tparam derivatives_type Type representing the derivative of the q-function w.r.t. its input arguments
 *
 * @note lambda does not appear as a template argument, as the directional derivative is
 * inherently just a linear transformation
 *
 * @tparam dsolution_type element solution
 * @tparam dresidual_type element residual
 *
 * @param[in] e The element number
 * @param[in] q The quadrature point index
 * @param[in] dU The element DOF values (primary input)
 * @param[inout] dR The element residuals (primary output)
 * @param[in] derivatives_ptr The address at which derivatives of the q-function with
 * respect to its arguments are stored
 * @param[in] J_ The Jacobians of the element transformations at all quadrature points
 * @see mfem::GeometricFactors
 * @param[in] num_elements The number of elements in the mesh
 */

template <Geometry g, typename test, typename trial, int Q, typename derivatives_type, typename du_elem_type,
          typename dr_elem_type>
SERAC_HOST_DEVICE void gradient_quadrature(int e, int q, du_elem_type& du_elem, dr_elem_type& dr_elem,
                                           derivatives_type*                         derivatives_ptr,
                                           const mfem::DeviceTensor<4, const double> J, int num_elements)
{
  using test_element         = finite_element<g, test>;
  using trial_element        = finite_element<g, trial>;
  static constexpr auto rule = GaussQuadratureRule<g, Q>();
  static constexpr int  dim  = dimension_of(g);

  // get the position of this quadrature point in the parent and physical space,
  // and calculate the measure of that point in physical space.
  auto   xi  = rule.points[q];
  auto   dxi = rule.weights[q];
  auto   J_q = make_tensor<dim, dim>([&](int i, int j) { return J(q, i, j, e); });
  double dx  = det(J_q) * dxi;

  // evaluate the (change in) value/derivatives at this quadrature point
  auto darg = Preprocess<trial_element>(du_elem, xi, J_q);

  // recall the derivative of the q-function w.r.t. its arguments at this quadrature point
  auto dq_darg = detail::AccessDerivatives(derivatives_ptr, e, q, rule, num_elements);

  // use the chain rule to compute the first-order change in the q-function output
  auto dq = chain_rule(dq_darg, darg);

  // integrate dq against test space shape functions / gradients
  // to get the (change in) element residual contributions
  dr_elem += Postprocess<test_element>(dq, xi, J_q) * dx;
}

}  // namespace domain_integral

}  // namespace serac

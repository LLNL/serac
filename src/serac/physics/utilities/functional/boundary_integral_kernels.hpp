// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include "serac/physics/utilities/quadrature_data.hpp"
#include "serac/physics/utilities/functional/integral_utilities.hpp"

namespace serac {

namespace boundary_integral {

/**
 * @overload
 * @note This specialization of detail::Preprocess is called when doing integrals
 * where the spatial dimension is different from the dimension of the element geometry
 * (i.e. surface integrals in 3D space, line integrals in 2D space, etc)
 *
 * TODO: provide gradients as well (needs some more info from mfem)
 */
template <typename element_type, typename T, typename coord_type>
auto Preprocess(const T& u, const coord_type& xi)
{
  if constexpr (element_type::family == Family::H1 || element_type::family == Family::L2) {
    return dot(u, element_type::shape_functions(xi));
  }

  // we can't support HCURL until some issues in mfem are fixed
  // if constexpr (element_type::family == Family::HCURL) {
  //  return dot(u, dot(element_type::shape_functions(xi), inv(J)));
  //}
}

/**
 * @overload
 * @note This specialization of detail::Postprocess is called when doing integrals
 * where the spatial dimension is different from the dimension of the element geometry
 * (i.e. surface integrals in 3D space, line integrals in 2D space, etc)
 *
 * In this case, q-function outputs are only integrated against test space shape functions
 */
template <typename element_type, typename T, typename coord_type>
auto Postprocess(const T& f, [[maybe_unused]] const coord_type& xi)
{
  if constexpr (element_type::family == Family::H1 || element_type::family == Family::L2) {
    return outer(element_type::shape_functions(xi), f);
  }

  // we can't support HCURL until fixing some shortcomings in mfem
  // if constexpr (element_type::family == Family::HCURL) {
  //  return outer(element_type::shape_functions(xi), dot(inv(J), f));
  //}

  if constexpr (element_type::family == Family::QOI) {
    return f;
  }
}

/**
 * @brief The base kernel template used to create different finite element calculation routines
 *
 * @tparam test The type of the test function space
 * @tparam trial The type of the trial function space
 * The above spaces can be any combination of {H1, Hcurl, Hdiv (TODO), L2 (TODO)}
 *
 * Template parameters other than the test and trial spaces are used for customization + optimization
 * and are erased through the @p std::function members of @p BoundaryIntegral
 * @tparam g The shape of the element (only quadrilateral and hexahedron are supported at present)
 * @tparam geometry_dim The dimension of the element (2 for quad, 3 for hex, etc)
 * @tparam spatial_dim The full dimension of the mesh
 * @tparam Q Quadrature parameter describing how many points per dimension
 * @tparam derivatives_type Type representing the derivative of the q-function (see below) w.r.t. its input arguments
 * @tparam lambda The actual quadrature-function (either lambda function or functor object) to
 * be evaluated at each quadrature point.
 * @see https://libceed.readthedocs.io/en/latest/libCEEDapi/#theoretical-framework for additional
 * information on the idea behind a quadrature function and its inputs/outputs
 * @tparam qpt_data_type The type of the data to store for each quadrature point
 *
 * @param[in] U The full set of per-element DOF values (primary input)
 * @param[inout] R The full set of per-element residuals (primary output)
 * @param[out] derivatives_ptr The address at which derivatives of @a lambda with
 * respect to its arguments will be stored
 * @param[in] J_ The Jacobians of the element transformations at all quadrature points
 * @param[in] X_ The actual (not reference) coordinates of all quadrature points
 * @param[in] N_ The unit normals of all quadrature points
 * @see mfem::GeometricFactors
 * @param[in] num_elements The number of elements in the mesh
 * @param[in] qf The actual quadrature function, see @p lambda
 */
template <Geometry g, typename test, typename trial, int Q, typename derivatives_type, typename lambda,
          typename qpt_data_type = void>
void evaluation_kernel(const mfem::Vector& U, mfem::Vector& R, CPUView<derivatives_type, 2> qf_derivatives,
                       const mfem::Vector& J_, const mfem::Vector& X_, const mfem::Vector& N_, int num_elements,
                       lambda qf)

{
  using test_element               = finite_element<g, test>;
  using trial_element              = finite_element<g, trial>;
  using element_residual_type      = typename test_element::residual_type;
  static constexpr int  dim        = dimension_of(g);
  static constexpr int  test_ndof  = test_element::ndof;
  static constexpr int  trial_ndof = trial_element::ndof;
  static constexpr auto rule       = GaussQuadratureRule<g, Q>();

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto N = mfem::Reshape(N_.Read(), rule.size(), dim + 1, num_elements);
  auto X = mfem::Reshape(X_.Read(), rule.size(), dim + 1, num_elements);
  auto J = mfem::Reshape(J_.Read(), rule.size(), num_elements);
  auto u = detail::Reshape<trial>(U.Read(), trial_ndof, num_elements);
  auto r = detail::Reshape<test>(R.ReadWrite(), test_ndof, num_elements);

  // for each element in the domain
  for (int e = 0; e < num_elements; e++) {
    // get the DOF values for this particular element
    tensor u_elem = detail::Load<trial_element>(u, e);

    // this is where we will accumulate the element residual tensor
    element_residual_type r_elem{};

    // for each quadrature point in the element
    for (int q = 0; q < static_cast<int>(rule.size()); q++) {
      // get the position of this quadrature point in the parent and physical space,
      // and calculate the measure of that point in physical space.
      auto   xi  = rule.points[q];
      auto   dxi = rule.weights[q];
      auto   x_q = make_tensor<dim + 1>([&](int i) { return X(q, i, e); });  // Physical coords of qpt
      auto   n_q = make_tensor<dim + 1>([&](int i) { return N(q, i, e); });  // Physical coords of unit normal
      double dx  = J(q, e) * dxi;

      // evaluate the value/derivatives needed for the q-function at this quadrature point
      auto arg = Preprocess<trial_element>(u_elem, xi);

      // evaluate the user-specified constitutive model
      //
      // note: make_dual(arg) promotes those arguments to dual number types
      // so that qf_output will contain values and derivatives
      auto qf_output = qf(x_q, n_q, make_dual(arg));

      // integrate qf_output against test space shape functions / gradients
      // to get element residual contributions
      r_elem += Postprocess<test_element>(get_value(qf_output), xi) * dx;

      // here, we store the derivative of the q-function w.r.t. its input arguments
      //
      // this will be used by other kernels to evaluate gradients / adjoints / directional derivatives
      qf_derivatives(static_cast<size_t>(e), static_cast<size_t>(q)) = get_gradient(qf_output);
    }

    // once we've finished the element integration loop, write our element residuals
    // out to memory, to be later assembled into global residuals by mfem
    detail::Add(r, r_elem, e);
  }
}

/**
 * @brief The base kernel template used to create create custom directional derivative
 * kernels associated with finite element calculations
 *
 * @tparam test The type of the test function space
 * @tparam trial The type of the trial function space
 * The above spaces can be any combination of {H1, Hcurl, Hdiv (TODO), L2 (TODO)}
 *
 * Template parameters other than the test and trial spaces are used for customization + optimization
 * and are erased through the @p std::function members of @p BoundaryIntegral
 * @tparam g The shape of the element (only quadrilateral and hexahedron are supported at present)
 * @tparam Q Quadrature parameter describing how many points per dimension
 * @tparam derivatives_type Type representing the derivative of the q-function w.r.t. its input arguments
 *
 * @note lambda does not appear as a template argument, as the directional derivative is
 * inherently just a linear transformation
 *
 * @param[in] dU The full set of per-element DOF values (primary input)
 * @param[inout] dR The full set of per-element residuals (primary output)
 * @param[in] derivatives_ptr The address at which derivatives of the q-function with
 * respect to its arguments are stored
 * @param[in] J_ The Jacobians of the element transformations at all quadrature points
 * @see mfem::GeometricFactors
 * @param[in] num_elements The number of elements in the mesh
 */
template <Geometry g, typename test, typename trial, int Q, typename derivatives_type>
void action_of_gradient_kernel(const mfem::Vector& dU, mfem::Vector& dR, CPUView<derivatives_type, 2> qf_derivatives,
                               const mfem::Vector& J_, int num_elements)
{
  using test_element               = finite_element<g, test>;
  using trial_element              = finite_element<g, trial>;
  using element_residual_type      = typename test_element::residual_type;
  static constexpr int  test_ndof  = test_element::ndof;
  static constexpr int  trial_ndof = trial_element::ndof;
  static constexpr auto rule       = GaussQuadratureRule<g, Q>();

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto J  = mfem::Reshape(J_.Read(), rule.size(), num_elements);
  auto du = detail::Reshape<trial>(dU.Read(), trial_ndof, num_elements);
  auto dr = detail::Reshape<test>(dR.ReadWrite(), test_ndof, num_elements);

  // for each element in the domain
  for (int e = 0; e < num_elements; e++) {
    // get the (change in) values for this particular element
    tensor du_elem = detail::Load<trial_element>(du, e);

    // this is where we will accumulate the (change in) element residual tensor
    element_residual_type dr_elem{};

    // for each quadrature point in the element
    for (int q = 0; q < static_cast<int>(rule.size()); q++) {
      // get the position of this quadrature point in the parent and physical space,
      // and calculate the measure of that point in physical space.
      auto   xi  = rule.points[q];
      auto   dxi = rule.weights[q];
      double dx  = J(q, e) * dxi;

      // evaluate the (change in) value/derivatives at this quadrature point
      auto darg = Preprocess<trial_element>(du_elem, xi);

      // recall the derivative of the q-function w.r.t. its arguments at this quadrature point
      auto dq_darg = qf_derivatives(static_cast<size_t>(e), static_cast<size_t>(q));

      // use the chain rule to compute the first-order change in the q-function output
      auto dq = chain_rule(dq_darg, darg);

      // integrate dq against test space shape functions / gradients
      // to get the (change in) element residual contributions
      dr_elem += Postprocess<test_element>(dq, xi) * dx;
    }

    // once we've finished the element integration loop, write our element residuals
    // out to memory, to be later assembled into global residuals by mfem
    detail::Add(dr, dr_elem, e);
  }
}

/**
 * @brief The base kernel template used to create create custom element stiffness matrices
 * associated with finite element calculations
 *
 * @tparam test The type of the test function space
 * @tparam trial The type of the trial function space
 * The above spaces can be any combination of {H1, Hcurl, Hdiv (TODO), L2 (TODO)}
 *
 * Template parameters other than the test and trial spaces are used for customization + optimization
 * and are erased through the @p std::function members of @p BoundaryIntegral
 * @tparam g The shape of the element (only quadrilateral and hexahedron are supported at present)
 * @tparam Q Quadrature parameter describing how many points per dimension
 * @tparam derivatives_type Type representing the derivative of the q-function w.r.t. its input arguments
 *
 * @note lambda does not appear as a template argument, as the stiffness matrix is
 * inherently just a linear transformation
 *
 * @param[in] dk array for storing each element's gradient contributions
 * @param[in] derivatives_ptr The address at which derivatives of the q-function with
 * respect to its arguments are stored
 * @param[in] J_ The Jacobians of the element transformations at all quadrature points
 * @see mfem::GeometricFactors
 * @param[in] num_elements The number of elements in the mesh
 */
template <Geometry g, typename test, typename trial, int Q, typename derivatives_type>
void element_gradient_kernel(CPUView<double, 3> dk, CPUView<derivatives_type, 2> qf_derivatives, const mfem::Vector& J_,
                             int num_elements)
{
  using test_element               = finite_element<g, test>;
  using trial_element              = finite_element<g, trial>;
  static constexpr int  test_ndof  = test_element::ndof;
  static constexpr int  test_dim   = test_element::components;
  static constexpr int  trial_ndof = trial_element::ndof;
  static constexpr int  trial_dim  = trial_element::components;
  static constexpr auto rule       = GaussQuadratureRule<g, Q>();

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto J = mfem::Reshape(J_.Read(), rule.size(), num_elements);

  // for each element in the domain
  for (int e = 0; e < num_elements; e++) {
    tensor<double, test_ndof, trial_ndof, test_dim, trial_dim> K_elem{};

    // for each quadrature point in the element
    for (int q = 0; q < static_cast<int>(rule.size()); q++) {
      // get the position of this quadrature point in the parent and physical space,
      // and calculate the measure of that point in physical space.
      auto   xi_q  = rule.points[q];
      auto   dxi_q = rule.weights[q];
      double dx    = J(q, e) * dxi_q;

      // recall the derivative of the q-function w.r.t. its arguments at this quadrature point
      auto dq_darg = qf_derivatives(static_cast<size_t>(e), static_cast<size_t>(q));

      if constexpr (std::is_same<test, QOI>::value) {
        auto N = trial_element::shape_functions(xi_q);
        for (int j = 0; j < trial_ndof; j++) {
          K_elem[0][j] += dq_darg * N[j] * dx;
        }
      } else {
        auto M = test_element::shape_functions(xi_q);
        auto N = trial_element::shape_functions(xi_q);

        for (int i = 0; i < test_ndof; i++) {
          for (int j = 0; j < trial_ndof; j++) {
            K_elem[i][j] += M[i] * dq_darg * N[j] * dx;
          }
        }
      }
    }

    // once we've finished the element integration loop, write our element gradients
    // out to memory, to be later assembled into the global gradient by mfem
    //
    // Note: we "transpose" these values to get them into the layout that mfem expects
    // clang-format off
    for_loop<test_ndof, test_dim, trial_ndof, trial_dim>([e, &dk, &K_elem](int i, int j, int k, int l) {
      dk(static_cast<size_t>(e), static_cast<size_t>(i + test_ndof * j), static_cast<size_t>(k + trial_ndof * l)) += K_elem[i][k][j][l];
    });
    // clang-format on
  }
}

}  // namespace boundary_integral

}  // namespace serac

// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file integral.hpp
 *
 * @brief This file contains the Integral core of the functional
 */
#pragma once

#include "mfem.hpp"
#include "mfem/linalg/dtensor.hpp"

#include "serac/physics/utilities/functional/tensor.hpp"
#include "serac/physics/utilities/functional/quadrature.hpp"
#include "serac/physics/utilities/functional/finite_element.hpp"
#include "serac/physics/utilities/functional/tuple_arithmetic.hpp"

// For now, mfem's support for getting surface element information (dof values, jacobians, etc)
// is lacking, making it difficult to actually implement surface integrals on our end
//
// #define ENABLE_BOUNDARY_INTEGRALS

namespace serac {

namespace detail {

/**
 * @brief Wrapper for mfem::Reshape compatible with Serac's finite element space types
 * @tparam space A test or trial space
 * @param[in] u The raw data to reshape into an mfem::DeviceTensor
 * @param[in] n1 The first dimension to reshape into
 * @param[in] n2 The second dimension to reshape into
 * @see mfem::Reshape
 */
template <typename space>
auto Reshape(double* u, int n1, int n2)
{
  if constexpr (space::components == 1) {
    return mfem::Reshape(u, n1, n2);
  }
  if constexpr (space::components > 1) {
    return mfem::Reshape(u, n1, space::components, n2);
  }
};
/// @overload
template <typename space>
auto Reshape(const double* u, int n1, int n2)
{
  if constexpr (space::components == 1) {
    return mfem::Reshape(u, n1, n2);
  }
  if constexpr (space::components > 1) {
    return mfem::Reshape(u, n1, space::components, n2);
  }
};

/**
 * @brief Extracts the dof values for a particular element
 * @param[in] u The decomposed per-element DOFs, libCEED's E-vector
 * @param[in] e The index of the element to retrieve DOFs for
 * @note For the case of only 1 dof per node, detail::Load returns a tensor<double, ndof>
 */
template <int ndof>
inline auto Load(const mfem::DeviceTensor<2, const double>& u, int e)
{
  return make_tensor<ndof>([&u, e](int i) { return u(i, e); });
}
/**
 * @overload
 * @note For the case of multiple dofs per node, detail::Load returns a tensor<double, components, ndof>
 */
template <int ndof, int components>
inline auto Load(const mfem::DeviceTensor<3, const double>& u, int e)
{
  return make_tensor<components, ndof>([&u, e](int j, int i) { return u(i, j, e); });
}
/**
 * @overload
 * @note Intended to be used with Serac's finite element space types
 */
template <typename space, typename T>
auto Load(const T& u, int e)
{
  if constexpr (space::components == 1) {
    return detail::Load<space::ndof>(u, e);
  }
  if constexpr (space::components > 1) {
    return detail::Load<space::ndof, space::components>(u, e);
  }
};

/**
 * @brief Adds the contributions of the local residual to the global residual
 * @param[inout] r_global The full element-decomposed residual
 * @param[in] r_local The contributions to a residual from a single element
 * @param[in] e The index of the element whose residual is @a r_local
 */
template <int ndof>
void Add(const mfem::DeviceTensor<2, double>& r_global, tensor<double, ndof> r_local, int e)
{
  for (int i = 0; i < ndof; i++) {
    r_global(i, e) += r_local[i];
  }
}
/**
 * @overload
 * @note Used when each node has multiple DOFs
 */
template <int ndof, int components>
void Add(const mfem::DeviceTensor<3, double>& r_global, tensor<double, ndof, components> r_local, int e)
{
  for (int i = 0; i < ndof; i++) {
    for (int j = 0; j < components; j++) {
      r_global(i, j, e) += r_local[i][j];
    }
  }
}

/**
 * @brief Computes the arguments to be passed into the q-function (shape function evaluations)
 * By default:
 *  H1 family elements will compute {value, gradient}
 *  Hcurl family elements will compute {value, curl}
 *  TODO: Hdiv family elements will compute {value, divergence}
 *  TODO: L2 family elements will compute value
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
auto Preprocess(T u, const tensor<double, dim> xi, const tensor<double, dim, dim> J)
{
  if constexpr (element_type::family == Family::H1 || element_type::family == Family::L2) {
    return std::tuple{dot(u, element_type::shape_functions(xi)),
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
    return std::tuple{value, curl};
  }
}

/**
 * @overload
 * @note This specialization of detail::Preprocess is called when doing integrals
 * where the spatial dimension is different from the dimension of the element geometry
 * (i.e. surface integrals in 3D space, line integrals in 2D space, etc)
 *
 * QUESTION: are gradients useful in these cases or not?
 */
template <typename element_type, typename T, int geometry_dim, int spatial_dim>
auto Preprocess(T u, const tensor<double, geometry_dim> xi,
                [[maybe_unused]] const tensor<double, spatial_dim, geometry_dim> J)
{
  if constexpr (element_type::family == Family::H1) {
    return dot(u, element_type::shape_functions(xi));
  }

  if constexpr (element_type::family == Family::HCURL) {
    return dot(u, dot(element_type::shape_functions(xi), inv(J)));
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
 * TODO: L2 family elements integrate f against test space shape functions
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
auto Postprocess(T f, const tensor<double, dim> xi, const tensor<double, dim, dim> J)
{
  // TODO: Helpful static_assert about f being tuple or tuple-like for H1, hcurl, hdiv
  if constexpr (element_type::family == Family::H1 || element_type::family == Family::L2) {
    auto W     = element_type::shape_functions(xi);
    auto dW_dx = dot(element_type::shape_function_gradients(xi), inv(J));
    return outer(W, std::get<0>(f)) + dot(dW_dx, std::get<1>(f));
  }

  if constexpr (element_type::family == Family::HCURL) {
    auto W      = dot(element_type::shape_functions(xi), inv(J));
    auto curl_W = element_type::shape_function_curl(xi) / det(J);
    if constexpr (dim == 3) {
      curl_W = dot(curl_W, transpose(J));
    }
    return (W * std::get<0>(f) + curl_W * std::get<1>(f));
  }
}

/**
 * @overload
 * @note This specialization of detail::Postprocess is called when doing integrals
 * where the spatial dimension is different from the dimension of the element geometry
 * (i.e. surface integrals in 3D space, line integrals in 2D space, etc)
 *
 * In this case, q-function outputs are only integrated against test space shape functions
 * QUESTION: Should test function gradients be supported here or not?
 */
template <typename element_type, typename T, int geometry_dim, int spatial_dim>
auto Postprocess(T f, const tensor<double, geometry_dim> xi,
                 [[maybe_unused]] const tensor<double, spatial_dim, geometry_dim> J)
{
  if constexpr (element_type::family == Family::H1) {
    return outer(element_type::shape_functions(xi), f);
  }

  if constexpr (element_type::family == Family::HCURL) {
    return outer(element_type::shape_functions(xi), dot(inv(J), f));
  }
}

/**
 * @brief Takes in a Jacobian matrix and computes the
 * associated length / area / volume ratio of the transformation
 *
 * In general, this is given by sqrt(det(J^T * J)), but for the case
 * where J is square, this is equivalent to just det(J)
 *
 * @param[in] A The Jacobian to compute the ratio on
 */
template <int n>
auto Measure(const tensor<double, n, n>& A)
{
  return det(A);
}

/// @overload
template <int m, int n>
auto Measure(const tensor<double, m, n>& A)
{
  return ::sqrt(det(transpose(A) * A));
}

}  // namespace detail

/**
 * @brief The base kernel template used to create different finite element calculation routines
 *
 * @tparam test The type of the test function space
 * @tparam trial The type of the trial function space
 * The above spaces can be any combination of {H1, Hcurl, Hdiv (TODO), L2 (TODO)}
 *
 * Template parameters other than the test and trial spaces are used for customization + optimization
 * and are erased through the @p std::function members of @p Integral
 * @tparam g The shape of the element (only quadrilateral and hexahedron are supported at present)
 * @tparam geometry_dim The dimension of the element (2 for quad, 3 for hex, etc)
 * @tparam spatial_dim The full dimension of the mesh
 * @tparam Q Quadrature parameter describing how many points per dimension
 * @tparam derivatives_type Type representing the derivative of the q-function (see below) w.r.t. its input arguments
 * @tparam lambda The actual quadrature-function (either lambda function or functor object) to
 * be evaluated at each quadrature point.
 * @see https://libceed.readthedocs.io/en/latest/libCEEDapi/#theoretical-framework for additional
 * information on the idea behind a quadrature function and its inputs/outputs
 *
 * @param[in] U The full set of per-element DOF values (primary input)
 * @param[inout] R The full set of per-element residuals (primary output)
 * @param[out] derivatives_ptr The address at which derivatives of @a lambda with
 * respect to its arguments will be stored
 * @param[in] J_ The Jacobians of the element transformations at all quadrature points
 * @param[in] X_ The actual (not reference) coordinates of all quadrature points
 * @see mfem::GeometricFactors
 * @param[in] num_elements The number of elements in the mesh
 * @param[in] qf The actual quadrature function, see @p lambda
 */
template <Geometry g, typename test, typename trial, int geometry_dim, int spatial_dim, int Q,
          typename derivatives_type, typename lambda>
void evaluation_kernel(const mfem::Vector& U, mfem::Vector& R, derivatives_type* derivatives_ptr,
                       const mfem::Vector& J_, const mfem::Vector& X_, int num_elements, lambda qf)
{
  using test_element               = finite_element<g, test>;
  using trial_element              = finite_element<g, trial>;
  using element_residual_type      = typename trial_element::residual_type;
  static constexpr int  test_ndof  = test_element::ndof;
  static constexpr int  trial_ndof = trial_element::ndof;
  static constexpr auto rule       = GaussQuadratureRule<g, Q>();

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto X = mfem::Reshape(X_.Read(), rule.size(), spatial_dim, num_elements);
  auto J = mfem::Reshape(J_.Read(), rule.size(), spatial_dim, geometry_dim, num_elements);
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
      auto   x_q = make_tensor<spatial_dim>([&](int i) { return X(q, i, e); });  // Physical coords of qpt
      auto   J_q = make_tensor<spatial_dim, geometry_dim>([&](int i, int j) { return J(q, i, j, e); });
      double dx  = detail::Measure(J_q) * dxi;

      // evaluate the value/derivatives needed for the q-function at this quadrature point
      auto arg = detail::Preprocess<trial_element>(u_elem, xi, J_q);

      // evaluate the user-specified constitutive model
      //
      // note: make_dual(arg) promotes those arguments to dual number types
      // so that qf_output will contain values and derivatives
      auto qf_output = qf(x_q, make_dual(arg));

      // integrate qf_output against test space shape functions / gradients
      // to get element residual contributions
      r_elem += detail::Postprocess<test_element>(get_value(qf_output), xi, J_q) * dx;

      // here, we store the derivative of the q-function w.r.t. its input arguments
      //
      // this will be used by other kernels to evaluate gradients / adjoints / directional derivatives
      derivatives_ptr[e * int(rule.size()) + q] = get_gradient(qf_output);
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
 * and are erased through the @p std::function members of @p Integral
 * @tparam g The shape of the element (only quadrilateral and hexahedron are supported at present)
 * @tparam geometry_dim The dimension of the element (2 for quad, 3 for hex, etc)
 * @tparam spatial_dim The full dimension of the mesh
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
template <Geometry g, typename test, typename trial, int geometry_dim, int spatial_dim, int Q,
          typename derivatives_type>
void gradient_kernel(const mfem::Vector& dU, mfem::Vector& dR, derivatives_type* derivatives_ptr,
                     const mfem::Vector& J_, int num_elements)
{
  using test_element               = finite_element<g, test>;
  using trial_element              = finite_element<g, trial>;
  using element_residual_type      = typename trial_element::residual_type;
  static constexpr int  test_ndof  = test_element::ndof;
  static constexpr int  trial_ndof = trial_element::ndof;
  static constexpr auto rule       = GaussQuadratureRule<g, Q>();

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto J  = mfem::Reshape(J_.Read(), rule.size(), spatial_dim, geometry_dim, num_elements);
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
      auto   J_q = make_tensor<spatial_dim, geometry_dim>([&](int i, int j) { return J(q, i, j, e); });
      double dx  = detail::Measure(J_q) * dxi;

      // evaluate the (change in) value/derivatives at this quadrature point
      auto darg = detail::Preprocess<trial_element>(du_elem, xi, J_q);

      // recall the derivative of the q-function w.r.t. its arguments at this quadrature point
      auto dq_darg = derivatives_ptr[e * int(rule.size()) + q];

      // use the chain rule to compute the first-order change in the q-function output
      auto dq = chain_rule(dq_darg, darg);

      // integrate dq against test space shape functions / gradients
      // to get the (change in) element residual contributions
      dr_elem += detail::Postprocess<test_element>(dq, xi, J_q) * dx;
    }

    // once we've finished the element integration loop, write our element residuals
    // out to memory, to be later assembled into global residuals by mfem
    detail::Add(dr, dr_elem, e);
  }
}

/**
 * @brief The base kernel template used to compute tangent element entries that can be assembled
 * into a tangent matrix
 *
 * @tparam test The type of the test function space
 * @tparam trial The type of the trial function space
 * The above spaces can be any combination of {H1, Hcurl, Hdiv (TODO), L2 (TODO)}
 *
 * Template parameters other than the test and trial spaces are used for customization + optimization
 * and are erased through the @p std::function members of @p Integral
 * @tparam g The shape of the element (only quadrilateral and hexahedron are supported at present)
 * @tparam geometry_dim The dimension of the element (2 for quad, 3 for hex, etc)
 * @tparam spatial_dim The full dimension of the mesh
 * @tparam Q Quadrature parameter describing how many points per dimension
 * @tparam derivatives_type Type representing the derivative of the q-function w.r.t. its input arguments
 *
 *
 * @param[inout] K_e The full set of per-element element tangents [test_ndofs x test_dim, trial_ndofs x trial_dim]
 * @param[in] derivatives_ptr The address at which derivatives of the q-function with
 * respect to its arguments are stored
 * @param[in] J_ The Jacobians of the element transformations at all quadrature points
 * @see mfem::GeometricFactors
 * @param[in] num_elements The number of elements in the mesh
 */
template <Geometry g, typename test, typename trial, int geometry_dim, int spatial_dim, int Q,
          typename derivatives_type>
void gradient_matrix_kernel(mfem::Vector& K_e, derivatives_type* derivatives_ptr, const mfem::Vector& J_,
                            int num_elements)
{
  using test_element  = finite_element<g, test>;
  using trial_element = finite_element<g, trial>;
  //  using element_residual_type      = typename trial_element::residual_type;
  static constexpr int                  test_ndof        = test_element::ndof;
  static constexpr int                  test_dim         = test_element::components;
  static constexpr int                  trial_ndof       = trial_element::ndof;
  static constexpr int                  trial_dim        = test_element::components;
  static constexpr auto                 rule             = GaussQuadratureRule<g, Q>();
  [[maybe_unused]] static constexpr int curl_spatial_dim = spatial_dim == 3 ? 3 : 1;

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto J  = mfem::Reshape(J_.Read(), rule.size(), spatial_dim, geometry_dim, num_elements);
  auto dk = mfem::Reshape(K_e.ReadWrite(), test_ndof * test_dim, trial_ndof * trial_dim, num_elements);

  // for each element in the domain
  for (int e = 0; e < num_elements; e++) {
    tensor<double, test_ndof * test_dim, trial_ndof * trial_dim> K_elem{};

    // for each quadrature point in the element
    for (int q = 0; q < static_cast<int>(rule.size()); q++) {
      // get the position of this quadrature point in the parent and physical space,
      // and calculate the measure of that point in physical space.
      auto                    xi_q  = rule.points[q];
      auto                    dxi_q = rule.weights[q];
      auto                    J_q = make_tensor<spatial_dim, geometry_dim>([&](int i, int j) { return J(q, i, j, e); });
      auto                    detJ_q = detail::Measure(J_q);
      [[maybe_unused]] double dx     = detJ_q * dxi_q;

      // recall the derivative of the q-function w.r.t. its arguments at this quadrature point
      auto dq_darg = derivatives_ptr[e * int(rule.size()) + q];

      // evaluate shape functions
      [[maybe_unused]] auto M = test_element::shape_functions(xi_q);
      [[maybe_unused]] auto N = trial_element::shape_functions(xi_q);
      if constexpr (test_element::family == Family::HCURL) {
        M = dot(M, inv(J_q));
      }
      if constexpr (trial_element::family == Family::HCURL) {
        N = dot(N, inv(J_q));
      }

      auto f00 = std::get<0>(std::get<0>(dq_darg));
      auto f01 = std::get<1>(std::get<0>(dq_darg));
      auto f10 = std::get<0>(std::get<1>(dq_darg));
      auto f11 = std::get<1>(std::get<1>(dq_darg));

      // df0_du stiffness contribution
      // size(M) = test_ndof
      // size(N) = trial_ndof
      // size(df0_du) = test_dim x trial_dim
      if constexpr (!is_zero<decltype(f00)>::value) {
        auto df0_du = convert_to_tensor_with_shape<test_dim, trial_dim>(f00);
        for_loop<test_ndof, test_dim, trial_ndof, trial_dim>([&](auto i, auto id, auto j, auto jd) {
          // maybe we should have a mapping for dofs x dim
          K_elem[i + test_ndof * id][j + trial_ndof * jd] += M[i] * df0_du[id][jd] * N[j] * dx;
        });
      }

      if constexpr (test_element::family == Family::H1 || test_element::family == Family::L2) {
        [[maybe_unused]] auto dM_dx = dot(test_element::shape_function_gradients(xi_q), inv(J_q));
        [[maybe_unused]] auto dN_dx = dot(trial_element::shape_function_gradients(xi_q), inv(J_q));

        // df0_dgradu stiffness contribution
        // size(M) = test_ndof
        // size(df0_dgradu) = test_dim x trial_dim x spatial_dim
        // size(dN_dx) = trial_ndof x spatial_dim
        if constexpr (!is_zero<decltype(f01)>::value) {
          auto df0_dgradu = convert_to_tensor_with_shape<test_dim, trial_dim, spatial_dim>(f01);
          for_loop<test_ndof, test_dim, trial_ndof, trial_dim, spatial_dim>(
              [&](auto i, auto id, auto j, auto jd, auto dummy_i) {
                // maybe we should have a mapping for dofs x dim
                K_elem[i + test_ndof * id][j + trial_ndof * jd] +=
                    M[i] * df0_dgradu[id][jd][dummy_i] * dN_dx[j][dummy_i] * dx;
              });
        }

        // // df1_du stiffness contribution
        // size(dM_dx) = test_ndof x spatial_dim
        // size(N) = trial_ndof
        // size(df1_du) = test_dim x spatial_dim x trial_dim
        if constexpr (!is_zero<decltype(f10)>::value) {
          auto& df1_du = f10;
          for_loop<test_ndof, test_dim, trial_ndof, trial_dim, spatial_dim>(
              [&](auto i, [[maybe_unused]] auto id, auto j, [[maybe_unused]] auto jd, auto dummy_i) {
                // maybe we should have a mapping for dofs x dim
                if constexpr (test_dim == 1 && trial_dim == 1) {
                  K_elem[i * test_dim][j * trial_dim] += dM_dx[i][dummy_i] * df1_du[dummy_i] * N[j] * dx;
                } else {
                  K_elem[i + test_ndof * id][j + trial_ndof * jd] +=
                      dM_dx[i][dummy_i] * df1_du[id][dummy_i][jd] * N[j] * dx;
                }
              });
        }

        // df1_dgradu stiffness contribution
        // size(dM_dx) = test_ndof x spatial_dim
        // size(dN_dx) = trial_ndof x spatial_dim
        // size(df1_dgradu) = test_dim x spatial_dim x trial_dim x spatial_dim
        if constexpr (!is_zero<decltype(f11)>::value) {
          auto& df1_dgradu = f11;
          for_loop<test_ndof, test_dim, trial_ndof, trial_dim, spatial_dim, spatial_dim>(
              [&](auto i, [[maybe_unused]] auto id, auto j, [[maybe_unused]] auto jd, auto dummy_i, auto dummy_j) {
                if constexpr (test_dim == 1 && trial_dim == 1) {
                  K_elem[i][j] += dM_dx[i][dummy_i] * df1_dgradu[dummy_i][dummy_j] * dN_dx[j][dummy_j] * dx;
                } else {
                  // maybe we should have a mapping for dofs x dim
                  K_elem[i + test_ndof * id][j + trial_ndof * jd] +=
                      dM_dx[i][dummy_i] * df1_dgradu[id][dummy_i][jd][dummy_j] * dN_dx[j][dummy_j] * dx;
                }
              });
        }
      } else {  // HCurl

        [[maybe_unused]] auto curl_M = test_element::shape_function_curl(xi_q) / det(J_q);
        if constexpr (spatial_dim == 3) {
          curl_M = dot(curl_M, transpose(J_q));
        }

        [[maybe_unused]] auto curl_N = trial_element::shape_function_curl(xi_q) / det(J_q);
        if constexpr (spatial_dim == 3) {
          curl_N = dot(curl_N, transpose(J_q));
        }

        // df0_dgradu stiffness contribution
        // size(M) = test_ndof
        // size(df0_dcurlu) = test_dim x trial_dim x curl_spatial_dim
        // size(curl_N) = trial_ndof x curl_spatial_dim
        if constexpr (!is_zero<decltype(f01)>::value) {
          auto df0_dcurlu = convert_to_tensor_with_shape<test_dim, trial_dim, curl_spatial_dim>(f01);
          for_loop<test_ndof, test_dim, trial_ndof, trial_dim, curl_spatial_dim>(
              [&](auto i, auto id, auto j, auto jd, [[maybe_unused]] auto dummy_i) {
                // maybe we should have a mapping for dofs x dim
                if constexpr (spatial_dim == 3) {
                  K_elem[i + test_ndof * id][j + trial_ndof * jd] +=
                      M[i] * df0_dcurlu[id][jd][dummy_i] * curl_N[j][dummy_i] * dx;
                } else {
                  K_elem[i + test_ndof * id][j + trial_ndof * jd] += M[i] * df0_dcurlu[id][jd] * curl_N[j] * dx;
                }
              });
        }

        // // df1_du stiffness contribution
        // size(curl_M) = test_ndof x curl_spatial_dim
        // size(N) = trial_ndof
        // size(df1_du) = test_dim x curl_spatial_dim x trial_dim
        if constexpr (!is_zero<decltype(f10)>::value) {
          auto& df1_du = f10;
          for_loop<test_ndof, test_dim, trial_ndof, trial_dim, curl_spatial_dim>(
              [&](auto i, [[maybe_unused]] auto id, auto j, [[maybe_unused]] auto jd, auto dummy_i) {
                // maybe we should have a mapping for dofs x dim
                if constexpr (test_dim == 1 && trial_dim == 1) {
                  if constexpr (spatial_dim == 3) {
                    K_elem[i * test_dim][j * trial_dim] += curl_M[i][dummy_i] * df1_du[dummy_i] * N[j] * dx;
                  } else {
                    K_elem[i * test_dim][j * trial_dim] += curl_M[i] * df1_du * N[j] * dx;
                  }
                } else {
                  if constexpr (spatial_dim == 3) {
                    K_elem[i + test_ndof * id][j + trial_ndof * jd] +=
                        curl_M[i][dummy_i] * df1_du[id][dummy_i][jd] * N[j] * dx;
                  } else {
                    K_elem[i + test_ndof * id][j + trial_ndof * jd] += curl_M[i] * df1_du[id][jd] * N[j] * dx;
                  }
                }
              });
        }

        // df1_dcurlu stiffness contribution
        // size(curl_M) = test_ndof x curl_spatial_dim
        // size(curl_N) = trial_ndof x curl_spatial_dim
        // size(df1_dcurl) = test_dim x curl_spatial_dim x trial_dim x curl_spatial_dim
        if constexpr (!is_zero<decltype(f11)>::value) {
          auto& df1_dcurlu = f11;
          for_loop<test_ndof, test_dim, trial_ndof, trial_dim, curl_spatial_dim, curl_spatial_dim>(
              [&](auto i, [[maybe_unused]] auto id, auto j, [[maybe_unused]] auto jd, [[maybe_unused]] auto dummy_i,
                  [[maybe_unused]] auto dummy_j) {
                // maybe we should have a mapping for dofs x dim
                if constexpr (test_dim == 1 && trial_dim == 1) {
                  if constexpr (spatial_dim == 3) {
                    K_elem[i][j] += curl_M[i][dummy_i] * df1_dcurlu[dummy_i][dummy_j] * curl_N[j][dummy_j] * dx;
                  } else {
                    K_elem[i][j] += curl_M[i] * df1_dcurlu * curl_N[j] * dx;
                  }
                } else {
                  if constexpr (spatial_dim == 3) {
                    K_elem[i + test_ndof * id][j + trial_ndof * jd] +=
                        curl_M[i][dummy_i] * df1_dcurlu[id][dummy_i][jd][dummy_j] * curl_N[j][dummy_j] * dx;
                  } else {
                    K_elem[i + test_ndof * id][j + trial_ndof * jd] += curl_M[i] * df1_dcurlu[id][jd] * curl_N[j] * dx;
                  }
                }
              });
        }
      }
    }

    // once we've finished the element integration loop, write our element stifness
    // out to memory, to be later assembled into global stifnesss by mfem
    for (int i = 0; i < test_ndof * test_dim; i++) {
      for (int j = 0; j < trial_ndof * trial_dim; j++) {
        dk(i, j, e) += K_elem[i][j];
      }
    }
  }
}

/// @cond
namespace detail {

/**
 * @brief a class that helps to extract the test space from a function signature template parameter
 * @tparam space The function signature itself
 */
template <typename spaces>
struct get_test_space;  // undefined

/**
 * @brief a class that helps to extract the test space from a function signature template parameter
 * @tparam space The function signature itself
 */
template <typename test_space, typename trial_space>
struct get_test_space<test_space(trial_space)> {
  using type = test_space;  ///< the test space
};

/**
 * @brief a class that helps to extract the trial space from a function signature template parameter
 * @tparam space The function signature itself
 */
template <typename spaces>
struct get_trial_space;  // undefined

/**
 * @brief a class that helps to extract the trial space from a function signature template parameter
 * @tparam space The function signature itself
 */
template <typename test_space, typename trial_space>
struct get_trial_space<test_space(trial_space)> {
  using type = trial_space;  ///< the trial space
};

/**
 * @brief a class that provides the lambda argument types for a given integral
 * @tparam trial_space the trial space associated with the integral
 * @tparam geometry_dim the dimensionality of the element type
 * @tparam spatial_dim the dimensionality of the space the mesh lives in
 */
template <typename space, int geometry_dim, int spatial_dim>
struct lambda_argument;

// specialization for an H1 space with polynomial order p, and c components
template <int p, int c, int dim>
struct lambda_argument<H1<p, c>, dim, dim> {
  using type = std::tuple<reduced_tensor<double, c>, reduced_tensor<double, c, dim> >;
};

// specialization for an L2 space with polynomial order p, and c components
template <int p, int c, int dim>
struct lambda_argument<L2<p, c>, dim, dim> {
  using type = std::tuple<reduced_tensor<double, c>, reduced_tensor<double, c, dim> >;
};

// specialization for an H1 space with polynomial order p, and c components
// evaluated in a line integral or surface integral. Note: only values are provided in this case
template <int p, int c, int geometry_dim, int spatial_dim>
struct lambda_argument<H1<p, c>, geometry_dim, spatial_dim> {
  using type = reduced_tensor<double, c>;
};

// specialization for an Hcurl space with polynomial order p in 2D
template <int p>
struct lambda_argument<Hcurl<p>, 2, 2> {
  using type = std::tuple<tensor<double, 2>, double>;
};

// specialization for an Hcurl space with polynomial order p in 3D
template <int p>
struct lambda_argument<Hcurl<p>, 3, 3> {
  using type = std::tuple<tensor<double, 3>, tensor<double, 3> >;
};

}  // namespace detail
/// @endcond

/**
 * @brief a type function that extracts the test space from a function signature template parameter
 * @tparam space The function signature itself
 */
template <typename spaces>
using test_space_t = typename detail::get_test_space<spaces>::type;

/**
 * @brief a type function that extracts the trial space from a function signature template parameter
 * @tparam space The function signature itself
 */
template <typename spaces>
using trial_space_t = typename detail::get_trial_space<spaces>::type;

static constexpr Geometry supported_geometries[] = {Geometry::Point, Geometry::Segment, Geometry::Quadrilateral,
                                                    Geometry::Hexahedron};

/**
 * @brief Describes a single integral term in a weak forumulation of a partial differential equation
 * @tparam spaces A @p std::function -like set of template parameters that describe the test and trial
 * function spaces, i.e., @p test(trial)
 */
template <typename spaces>
class Integral {
public:
  using test_space  = test_space_t<spaces>;   ///< the test function space
  using trial_space = trial_space_t<spaces>;  ///< the trial function space

  /**
   * @brief Constructs an @p Integral from a user-provided quadrature function
   * @tparam geometry_dim The dimension of the element (2 for quad, 3 for hex, etc)
   * @tparam spatial_dim The full dimension of the mesh
   * @param[in] num_elements The number of elements in the mesh
   * @param[in] J The Jacobians of the element transformations at all quadrature points
   * @param[in] X The actual (not reference) coordinates of all quadrature points
   * @see mfem::GeometricFactors
   * @param[in] qf The user-provided quadrature function
   * @note The @p Dimension parameters are used to assist in the deduction of the @a geometry_dim
   * and @a spatial_dim template parameters
   */
  template <int geometry_dim, int spatial_dim, typename lambda_type>
  Integral(int num_elements, const mfem::Vector& J, const mfem::Vector& X, Dimension<geometry_dim>,
           Dimension<spatial_dim>, lambda_type&& qf)
      : J_(J), X_(X)
  {
    constexpr auto geometry                      = supported_geometries[geometry_dim];
    constexpr auto Q                             = std::max(test_space::order, trial_space::order) + 1;
    constexpr auto quadrature_points_per_element = (spatial_dim == 2) ? Q * Q : Q * Q * Q;

    uint32_t num_quadrature_points = quadrature_points_per_element * uint32_t(num_elements);

    // these lines of code figure out the argument types that will be passed
    // into the quadrature function in the finite element kernel.
    //
    // we use them to observe the output type and allocate memory to store
    // the derivative information at each quadrature point
    using x_t             = tensor<double, spatial_dim>;
    using u_du_t          = typename detail::lambda_argument<trial_space, geometry_dim, spatial_dim>::type;
    using derivative_type = decltype(get_gradient(qf(x_t{}, make_dual(u_du_t{}))));

    // the derivative_type data is stored in a shared_ptr here, because it can't be a
    // member variable on the Integral class template (since it depends on the lambda function,
    // which isn't known until the time of construction).
    //
    // This shared_ptr should have a comparable lifetime to the Integral instance itself, since
    // the reference count will increase when it is captured by the lambda functions below, and
    // the reference count will go back to zero after those std::functions are deconstructed in
    // Integral::~Integral()
    //
    // derivatives are stored as a 2D array, such that quadrature point q of element e is accessed by
    // qf_derivatives[e * quadrature_points_per_element + q]
    std::shared_ptr<derivative_type[]> qf_derivatives(new derivative_type[num_quadrature_points]);

    // this is where we actually specialize the finite element kernel templates with
    // our specific requirements (element type, test/trial spaces, quadrature rule, q-function, etc).
    //
    // std::function's type erasure lets us wrap those specific details inside a function with known signature
    //
    // note: the qf_derivatives_ptr is copied by value to each lambda function below,
    //       to allow the evaluation kernel to pass derivative values to the gradient kernel
    evaluation_ = [=](const mfem::Vector& U, mfem::Vector& R) {
      evaluation_kernel<geometry, test_space, trial_space, geometry_dim, spatial_dim, Q>(U, R, qf_derivatives.get(), J_,
                                                                                         X_, num_elements, qf);
    };

    gradient_ = [=](const mfem::Vector& dU, mfem::Vector& dR) {
      gradient_kernel<geometry, test_space, trial_space, geometry_dim, spatial_dim, Q>(dU, dR, qf_derivatives.get(), J_,
                                                                                       num_elements);
    };

    gradient_mat_ = [=](mfem::Vector& K_e) {
      gradient_matrix_kernel<geometry, test_space, trial_space, geometry_dim, spatial_dim, Q>(K_e, qf_derivatives.get(),
                                                                                              J_, num_elements);
    };
  }

  /**
   * @brief Applies the integral, i.e., @a output_E = evaluate( @a input_E )
   * @param[in] input_E The input to the evaluation; per-element DOF values
   * @param[out] output_E The output of the evalution; per-element DOF residuals
   * @see evaluation_kernel
   */
  void Mult(const mfem::Vector& input_E, mfem::Vector& output_E) const { evaluation_(input_E, output_E); }

  /**
   * @brief Applies the integral, i.e., @a output_E = gradient( @a input_E )
   * @param[in] input_E The input to the evaluation; per-element DOF values
   * @param[out] output_E The output of the evalution; per-element DOF residuals
   * @see gradient_kernel
   */
  void GradientMult(const mfem::Vector& input_E, mfem::Vector& output_E) const { gradient_(input_E, output_E); }

  /**
   * @brief Computes the element stiffness matrices, storing them in an `mfem::Vector` that has been reshaped into a
   * multidimensional array
   * @param[inout] K_e The reshaped vector as a mfem::DeviceTensor of size (test_dim * test_dof, trial_dim * trial_dof,
   * elem)
   */
  void ComputeElementMatrices(mfem::Vector& K_e) const { gradient_mat_(K_e); }

private:
  /**
   * @brief Jacobians of the element transformations at all quadrature points
   */
  const mfem::Vector J_;
  /**
   * @brief Mapped (physical) coordinates of all quadrature points
   */
  const mfem::Vector X_;

  /**
   * @brief Type-erased handle to evaluation kernel
   * @see evaluation_kernel
   */
  std::function<void(const mfem::Vector&, mfem::Vector&)> evaluation_;
  /**
   * @brief Type-erased handle to gradient kernel
   * @see gradient_kernel
   */
  std::function<void(const mfem::Vector&, mfem::Vector&)> gradient_;
  /**
   * @brief Type-erased handle to gradient matrix assembly kernel
   * @see gradient_matrix_kernel
   */
  std::function<void(mfem::Vector&)> gradient_mat_;
};

}  // namespace serac

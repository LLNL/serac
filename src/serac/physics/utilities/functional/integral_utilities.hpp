// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file intergal_utilities.hpp
 *
 * @brief this file contains functions and tools used by both domain_integral.hpp and boundary_integral.hpp
 */

#pragma once

#include "mfem.hpp"
#include "mfem/linalg/dtensor.hpp"
//#include "mfem/general/backends.hpp"

#include "serac/physics/utilities/functional/tensor.hpp"
#include "serac/physics/utilities/functional/finite_element.hpp"

namespace serac {

namespace detail {

/**
 * @brief a constexpr function for computing an integer raised to an integer power
 * @param[in] x the (integer) number to be raised to some power
 * @param[in] n the (integer) power
 */
constexpr int pow(int x, int n)
{
  int x_to_the_n = 1;
  for (int i = 0; i < n; i++) {
    x_to_the_n *= x;
  }
  return x_to_the_n;
}

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
SERAC_HOST_DEVICE inline auto Load(const mfem::DeviceTensor<2, const double>& u, int e)
{
  return make_tensor<ndof>([&u, e](int i) { return u(i, e); });
}

/**
 * @overload
 * @note For the case of multiple dofs per node, detail::Load returns a tensor<double, components, ndof>
 */
template <int ndof, int components>
SERAC_HOST_DEVICE inline auto Load(const mfem::DeviceTensor<3, const double>& u, int e)
{
  return make_tensor<components, ndof>([&u, e](int j, int i) { return u(i, j, e); });
}

/**
 * @overload
 * @note Intended to be used with Serac's finite element space types
 */
template <typename space, typename T>
SERAC_HOST_DEVICE auto Load(const T& u, int e)
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
SERAC_HOST_DEVICE void Add(const mfem::DeviceTensor<2, double>& r_global, tensor<double, ndof> r_local, int e)
{
  for (int i = 0; i < ndof; i++) {
    // r_global(i, e) += r_local[i];
    AtomicAdd(r_global(i, e), r_local[i]);
  }
}

/**
 * @overload
 * @note Used when each node has multiple DOFs
 */
template <int ndof, int components>
SERAC_HOST_DEVICE void Add(const mfem::DeviceTensor<3, double>& r_global, tensor<double, ndof, components> r_local,
                           int e)
{
  for (int i = 0; i < ndof; i++) {
    for (int j = 0; j < components; j++) {
      r_global(i, j, e) += r_local[i][j];
    }
  }
}

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
  using type = serac::tuple<reduced_tensor<double, c>, reduced_tensor<double, c, dim> >;
};

// specialization for an L2 space with polynomial order p, and c components
template <int p, int c, int dim>
struct lambda_argument<L2<p, c>, dim, dim> {
  using type = serac::tuple<reduced_tensor<double, c>, reduced_tensor<double, c, dim> >;
};

// specialization for an H1 space with polynomial order p, and c components
// evaluated in a line integral or surface integral. Note: only values are provided in this case
template <int p, int c, int geometry_dim, int spatial_dim>
struct lambda_argument<H1<p, c>, geometry_dim, spatial_dim> {
  using type = reduced_tensor<double, c>;
};

// specialization for an H1 space with polynomial order p, and c components
// evaluated in a line integral or surface integral. Note: only values are provided in this case
template <int p, int c, int geometry_dim, int spatial_dim>
struct lambda_argument<L2<p, c>, geometry_dim, spatial_dim> {
  using type = reduced_tensor<double, c>;
};

// specialization for an Hcurl space with polynomial order p in 2D
template <int p>
struct lambda_argument<Hcurl<p>, 2, 2> {
  using type = serac::tuple<tensor<double, 2>, double>;
};

// specialization for an Hcurl space with polynomial order p in 3D
template <int p>
struct lambda_argument<Hcurl<p>, 3, 3> {
  using type = serac::tuple<tensor<double, 3>, tensor<double, 3> >;
};

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
 * @overload
 * @note This specialization of detail::Preprocess is called when doing integrals
 * where the spatial dimension is different from the dimension of the element geometry
 * (i.e. surface integrals in 3D space, line integrals in 2D space, etc)
 *
 * QUESTION: are gradients useful in these cases or not?
 */
template <typename element_type, typename T, int geometry_dim, int spatial_dim>
SERAC_HOST_DEVICE auto Preprocess(T u, const tensor<double, geometry_dim> xi,
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
 *  H1 family elements integrate serac::get<0>(f) against the test space shape functions
 *                           and serac::get<1>(f) against the test space shape function gradients
 *  Hcurl family elements integrate serac::get<0>(f) against the test space shape functions
 *                              and serac::get<1>(f) against the curl of the test space shape functions
 * TODO: Hdiv family elements integrate serac::get<0>(f) against the test space shape functions
 *                                  and serac::get<1>(f) against the divergence of the test space shape functions
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
 * @overload
 * @note This specialization of detail::Postprocess is called when doing integrals
 * where the spatial dimension is different from the dimension of the element geometry
 * (i.e. surface integrals in 3D space, line integrals in 2D space, etc)
 *
 * In this case, q-function outputs are only integrated against test space shape functions
 * QUESTION: Should test function gradients be supported here or not?
 */
template <typename element_type, typename T, int geometry_dim, int spatial_dim>
SERAC_HOST_DEVICE auto Postprocess(T f, const tensor<double, geometry_dim> xi,
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
SERAC_HOST_DEVICE auto Measure(const tensor<double, n, n>& A)
{
  return det(A);
}

/// @overload
template <int m, int n>
SERAC_HOST_DEVICE auto Measure(const tensor<double, m, n>& A)
{
  return ::sqrt(det(transpose(A) * A));
}

/**
 * @brief derivatives_ptr access
 *
 * Templating this will allow us to change the stride-access patterns more consistently
 *
 * @param[in] derivative_ptr pointer to derivatives
 * @param[in] e element number
 * @param[in] q qaudrature number
 * @param[in] rule quadrature rule
 * @param[in] num_elements number of finite elements
 */
template <typename derivatives_type, typename rule_type, bool quadrature_coalescing = true>
SERAC_HOST_DEVICE derivatives_type& AccessDerivatives(derivatives_type* derivatives_ptr, int e, int q,
                                                      [[maybe_unused]] rule_type& rule,
                                                      [[maybe_unused]] int        num_elements)
{
  if constexpr (quadrature_coalescing) {
    return derivatives_ptr[e * int(rule.size()) + q];
  } else {
    return derivatives_ptr[q * num_elements + e];
  }
}

}  // namespace detail

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

}  // namespace serac

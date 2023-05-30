// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file integral_utilities.hpp
 *
 * @brief this file contains functions and tools used by both domain_integral.hpp and boundary_integral.hpp
 */

#pragma once

#include "mfem.hpp"
#include "mfem/linalg/dtensor.hpp"

#include "serac/numerics/functional/quadrature_data.hpp"
#include "serac/numerics/functional/tuple.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/quadrature.hpp"

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
 * @brief a class that provides the lambda argument types for a given integral
 * @tparam trial_space the trial space associated with the integral
 * @tparam geometry_dim the dimensionality of the element type
 * @tparam spatial_dim the dimensionality of the space the mesh lives in
 */
template <typename space, int geometry_dim, int spatial_dim>
struct lambda_argument;

/**
 * @overload
 * @note specialization for an H1 space with polynomial order p, and c components
 */
template <int p, int c, int dim>
struct lambda_argument<H1<p, c>, dim, dim> {
  /**
   * @brief The arguments for the lambda function
   */
  using type = serac::tuple<reduced_tensor<double, c>, reduced_tensor<double, c, dim>>;
};

/**
 * @overload
 * @note specialization for an L2 space with polynomial order p, and c components
 */
template <int p, int c, int dim>
struct lambda_argument<L2<p, c>, dim, dim> {
  /**
   * @brief The arguments for the lambda function
   */
  using type = serac::tuple<reduced_tensor<double, c>, reduced_tensor<double, c, dim>>;
};

/**
 * @overload
 * @note specialization for an H1 space with polynomial order p, and c components
 *       evaluated in a line integral or surface integral. Note: only values are provided in this case
 */
template <int p, int c, int geometry_dim, int spatial_dim>
struct lambda_argument<H1<p, c>, geometry_dim, spatial_dim> {
  /**
   * @brief The arguments for the lambda function
   */
  using type = reduced_tensor<double, c>;
};

/**
 * @overload
 * @note specialization for an H1 space with polynomial order p, and c components
 *       evaluated in a line integral or surface integral. Note: only values are provided in this case
 */
template <int p, int c, int geometry_dim, int spatial_dim>
struct lambda_argument<L2<p, c>, geometry_dim, spatial_dim> {
  /**
   * @brief The arguments for the lambda function
   */
  using type = reduced_tensor<double, c>;
};

/**
 * @overload
 * @note specialization for an Hcurl space with polynomial order p in 2D
 */
template <int p>
struct lambda_argument<Hcurl<p>, 2, 2> {
  /**
   * @brief The arguments for the lambda function
   */
  using type = serac::tuple<tensor<double, 2>, double>;
};

/**
 * @overload
 * @note specialization for an Hcurl space with polynomial order p in 3D
 */
template <int p>
struct lambda_argument<Hcurl<p>, 3, 3> {
  /**
   * @brief The arguments for the lambda function
   */
  using type = serac::tuple<tensor<double, 3>, tensor<double, 3>>;
};

/**
 * @brief Determines the return type of a qfunction lambda
 * @tparam lambda_type The type of the lambda itself
 * @tparam x_t The type of the "value" itself
 * @tparam u_du_t The type of the derivative
 * @tparam qpt_data_type The type of the per-quadrature state data, @p void when not applicable
 */
template <typename lambda_type, typename x_t, typename u_du_t, typename qpt_data_type, typename SFINAE = void>
struct qf_result {
  /**
   * @brief The type of the quadrature function result
   */
  using type = std::invoke_result_t<lambda_type, x_t, decltype(make_dual(std::declval<u_du_t>()))>;
};

/**
 * @overload
 */
template <typename lambda_type, typename x_t, typename u_du_t, typename qpt_data_type>
struct qf_result<lambda_type, x_t, u_du_t, qpt_data_type, std::enable_if_t<!std::is_same_v<qpt_data_type, void>>> {
  /**
   * @brief The type of the quadrature function result
   * @note Expecting that qf lambdas take an lvalue reference to a state
   */
  using type = std::invoke_result_t<lambda_type, x_t, decltype(make_dual(std::declval<u_du_t>())),
                                    std::add_lvalue_reference_t<qpt_data_type>>;
};

/**
 * @brief derivatives_ptr access
 *
 * Templating this will allow us to change the stride-access patterns more consistently
 * By default derivatives_ptr is accessed using row_major ordering derivatives_ptr(element, quadrature).
 *
 * @tparam derivatives_type The type of the derivatives
 * @tparam rule_type The type of the quadrature rule
 * @tparam row_major A boolean to choose to use row major access patterns or column major
 * @param[in] derivatives_ptr pointer to derivatives
 * @param[in] e element number
 * @param[in] q qaudrature number
 * @param[in] rule quadrature rule
 * @param[in] num_elements number of finite elements
 */
template <typename derivatives_type, typename rule_type, bool row_major = true>
SERAC_HOST_DEVICE constexpr derivatives_type& AccessDerivatives(derivatives_type* derivatives_ptr, int e, int q,
                                                                [[maybe_unused]] rule_type& rule,
                                                                [[maybe_unused]] int        num_elements)
{
  if constexpr (row_major) {
    return derivatives_ptr[e * static_cast<int>(rule.size()) + q];
  } else {
    return derivatives_ptr[q * num_elements + e];
  }
}

/// @brief layer of indirection needed to unpack the entries of the argument tuple
SERAC_SUPPRESS_NVCC_HOSTDEVICE_WARNING
template <typename lambda, typename coords_type, typename T, typename qpt_data_type, int... i>
SERAC_HOST_DEVICE auto apply_qf_helper(lambda&& qf, coords_type&& x_q, qpt_data_type&& qpt_data, const T& arg_tuple,
                                       std::integer_sequence<int, i...>)
{
  if constexpr (std::is_same<typename std::decay<qpt_data_type>::type, Nothing>::value) {
    return qf(x_q, serac::get<i>(arg_tuple)...);
  } else {
    return qf(x_q, qpt_data, serac::get<i>(arg_tuple)...);
  }
}

/// @overload
SERAC_SUPPRESS_NVCC_HOSTDEVICE_WARNING
template <typename lambda, typename coords_type, typename T, int... i>
SERAC_HOST_DEVICE auto apply_qf_helper(lambda&& qf, coords_type&& x_q, coords_type&& n_q, const T& arg_tuple,
                                       std::integer_sequence<int, i...>)
{
  return qf(x_q, n_q, serac::get<i>(arg_tuple)...);
}

/**
 * @brief Actually calls the q-function
 * This is an indirection layer to provide a transparent call site usage regardless of whether
 * quadrature point (state) information is required
 * @param[in] qf The quadrature function functor object
 * @param[in] x_q The physical coordinates of the quadrature point
 * @param[in] arg_tuple The values and derivatives at the quadrature point, as a dual
 * @param[inout] qpt_data The state information at the quadrature point
 */
template <typename lambda, typename coords_type, typename... T, typename qpt_data_type>
SERAC_HOST_DEVICE auto apply_qf(lambda&& qf, coords_type&& x_q, qpt_data_type&& qpt_data,
                                const serac::tuple<T...>& arg_tuple)
{
  return apply_qf_helper(qf, x_q, qpt_data, arg_tuple,
                         std::make_integer_sequence<int, static_cast<int>(sizeof...(T))>{});
}

/**
 * @overload
 * @note: boundary integrals pass the unit normal vector as second argument and do not support qpt_data
 */
template <typename lambda, typename coords_type, typename... T>
SERAC_HOST_DEVICE auto apply_qf(lambda&& qf, coords_type&& x_q, coords_type&& n_q, const serac::tuple<T...>& arg_tuple)
{
  return apply_qf_helper(qf, x_q, n_q, arg_tuple, std::make_integer_sequence<int, static_cast<int>(sizeof...(T))>{});
}

}  // namespace detail

static constexpr mfem::Geometry::Type supported_geometries[] = {mfem::Geometry::POINT, mfem::Geometry::SEGMENT,
                                                                mfem::Geometry::SQUARE, mfem::Geometry::CUBE};

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
 * @tparam element_type The type of the element (used to determine the family)
 * @tparam T the type of the element values to be interpolated and differentiated
 * @tparam dim the geometric dimension of the element
 *
 * @param[in] u The DOF values for the element
 * @param[in] xi The position of the quadrature point in reference space
 * @param[in] J The Jacobian of the element transformation at the quadrature point
 */
template <typename element_type, typename T, int dim>
SERAC_HOST_DEVICE auto Preprocess(T u, const tensor<double, dim>& xi, const tensor<double, dim, dim>& J)
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
 * @brief
 *
 * @tparam element_type The type of the element (used to determine the family)
 * @tparam T the type of the element values to be interpolated and differentiated
 * @tparam dim the geometric dimension of the element
 *
 * @param[in] u The DOF values for each element
 * @param[in] xi The position of the quadrature point in reference space
 * @param[in] J The Jacobian of the element transformation at the quadrature point
 */
template <mfem::Geometry::Type geom, typename... trials, typename tuple_type, int dim, int... i>
SERAC_HOST_DEVICE auto PreprocessHelper(const tuple_type& u, const tensor<double, dim>& xi,
                                        const tensor<double, dim, dim>& J, std::integer_sequence<int, i...>)
{
  return serac::make_tuple(Preprocess<finite_element<geom, trials>>(get<i>(u), xi, J)...);
}

/**
 * @overload
 * @note multi-trial space overload of Preprocess
 */
template <mfem::Geometry::Type geom, typename... trials, typename tuple_type, int dim>
SERAC_HOST_DEVICE auto Preprocess(const tuple_type& u, const tensor<double, dim>& xi, const tensor<double, dim, dim>& J)
{
  return PreprocessHelper<geom, trials...>(u, xi, J,
                                           std::make_integer_sequence<int, static_cast<int>(sizeof...(trials))>{});
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
SERAC_HOST_DEVICE auto Postprocess(const T& f, const tensor<double, dim>& xi, const tensor<double, dim, dim>& J)
{
  // TODO: Helpful static_assert about f being tuple or tuple-like for H1, hcurl, hdiv
  if constexpr (element_type::family == Family::QOI) {
    return f;
  }

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

}  // namespace domain_integral

}  // namespace serac

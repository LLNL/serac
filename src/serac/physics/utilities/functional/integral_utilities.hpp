// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
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

#include "serac/physics/utilities/functional/tuple.hpp"
#include "serac/physics/utilities/functional/tensor.hpp"
#include "serac/physics/utilities/functional/quadrature.hpp"
#include "serac/physics/utilities/functional/tuple_arithmetic.hpp"

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
 * @overload
 * @note Used for quantities of interest
 */
void Add(const mfem::DeviceTensor<2, double>& r_global, double r_local, int e) { r_global(0, e) += r_local; }

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
      AtomicAdd(r_global(i, j, e), r_local[i][j]);
    }
  }
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
 * @param[in] derivative_ptr pointer to derivatives
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
    return derivatives_ptr[e * int(rule.size()) + q];
  } else {
    return derivatives_ptr[q * num_elements + e];
  }
}

}  // namespace detail

static constexpr Geometry supported_geometries[] = {Geometry::Point, Geometry::Segment, Geometry::Quadrilateral,
                                                    Geometry::Hexahedron};

template <typename T1, typename T2>
struct linear_approximation {
  T1 value;
  T2 derivative;
};

/**
 * @brief a function that evaluates and packs the shape function information in a way
 * that makes the element gradient evaluation simpler to implement
 *
 * @tparam element_type which type of finite element shape functions to evaluate
 * @tparam dim the geometric dimension of the element
 *
 * @param[in] xi where to evaluate shape functions and their gradients
 * @param[in] J the jacobian, dx_dxi, of the isoparametric map from parent-to-physical coordinates
 */
template <typename element_type, int dim>
auto evaluate_shape_functions(const tensor<double, dim>& xi, const tensor<double, dim, dim>& J)
{
  if constexpr (element_type::family == Family::HCURL) {
    auto N      = dot(element_type::shape_functions(xi), inv(J));
    auto curl_N = element_type::shape_function_curl(xi) / det(J);
    if constexpr (dim == 3) {
      curl_N = dot(curl_N, transpose(J));
    }

    using pair_t =
        linear_approximation<std::remove_reference_t<decltype(N[0])>, std::remove_reference_t<decltype(curl_N[0])>>;
    tensor<pair_t, element_type::ndof> output{};
    for (int i = 0; i < element_type::ndof; i++) {
      output[i].value      = N[i];
      output[i].derivative = curl_N[i];
    }
    return output;
  }

  if constexpr (element_type::family == Family::H1 || element_type::family == Family::L2) {
    auto N      = element_type::shape_functions(xi);
    auto grad_N = dot(element_type::shape_function_gradients(xi), inv(J));

    using pair_t =
        linear_approximation<std::remove_reference_t<decltype(N[0])>, std::remove_reference_t<decltype(grad_N[0])>>;
    tensor<pair_t, element_type::ndof> output{};
    for (int i = 0; i < element_type::ndof; i++) {
      output[i].value      = N[i];
      output[i].derivative = grad_N[i];
    }
    return output;
  }
}

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

namespace detail {

/**
 * @brief a (poorly named) tuple of quantities used to discover the sparsity
 * pattern associated with element and boundary element matrices.
 *
 * It stores information about how the entries of an element "stiffness" matrix
 * map to the global stiffness. The operator< definition allows us to sort them
 * lexicographically, to facilitate creating the CSR matrix graph.
 */
struct ElemInfo {
  int  global_row;
  int  global_col;
  int  local_row;
  int  local_col;
  int  element_id;
  int  sign;
  bool on_boundary;
};

/**
 * @brief operator for sorting lexicographically by {global_row, global_col}
 * @param x the ElemInfo on the left
 * @param y the ElemInfo on the right
 */
inline bool operator<(const ElemInfo& x, const ElemInfo& y)
{
  return (x.global_row < y.global_row) || (x.global_row == y.global_row && x.global_col < y.global_col);
}

/**
 * @brief operator determining inequality by {global_row, global_col}
 * @param x the ElemInfo on the left
 * @param y the ElemInfo on the right
 */
inline bool operator!=(const ElemInfo& x, const ElemInfo& y)
{
  return (x.global_row != y.global_row) || (x.global_col != y.global_col);
}

/**
 * @brief mfem will frequently encode {sign, index} into a single int32_t.
 * This function decodes the sign from such a type.
 */
inline int get_sign(int i) { return (i >= 0) ? 1 : -1; }

/**
 * @brief mfem will frequently encode {sign, index} into a single int32_t.
 * This function decodes the index from such a type.
 */
inline int get_index(int i) { return (i >= 0) ? i : -1 - i; }

/**
 * @brief this type explicitly stores sign (typically used conveying edge/face orientation) and index values
 *
 * TODO: investigate implementation via bitfield (should have smaller memory footprint, better readability than mfem's
 * {sign, index} int32_t encoding)
 */
struct SignedIndex {
  int index;
  int sign;
      operator int() { return index; }
};

/**
 * @brief reorder the entries of an array of integers according to the given permutation array
 *
 * @param permutation the array describing how to reorder the input values
 * @param input the array to be reordered
 *
 * @note permutation[i] describes where input[i] will appear in the output array.
 *
 * @note if entry permutation[i] is negative, it will be interpreted as a reordering
 * (according to its index) and sign change of the permuted value (according to mfem convention (?))
 */
inline void apply_permutation(const mfem::Array<int>& permutation, mfem::Array<int>& input)
{
  auto output = input;
  for (int i = 0; i < permutation.Size(); i++) {
    if (permutation[i] >= 0) {
      output[i] = input[permutation[i]];
    } else {
      output[i] = -input[-permutation[i] - 1] - 1;
    }
  }
  input = output;
}

}  // namespace detail

}  // namespace serac

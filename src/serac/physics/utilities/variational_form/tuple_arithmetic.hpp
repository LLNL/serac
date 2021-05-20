// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file tuple_arithmetic.hpp
 *
 * @brief Definitions of arithmetic operations on tuples of values
 */

// this file defines basic arithmetic operations on tuples of values
// so that expressions like sum1 and sum2 below are equivalent
//
// std::tuple< foo, bar > a;
// std::tuple< baz, qux > b;
//
// std::tuple sum1 = a + b;
// std::tuple sum2{std::get<0>(a) + std::get<0>(b), std::get<1>(a) + std::get<1>(b)};

#pragma once

#include <tuple>

#include "tensor.hpp"

namespace serac {

namespace detail {

/**
 * @brief Trait for checking if a type is a @p std::tuple
 */
template <typename T>
struct is_tuple {
  // FIXME: Should we use std::false_type/std::true_type
  static constexpr bool value = false;
};
/// @overload
template <typename... T>
struct is_tuple<std::tuple<T...> > {
  static constexpr bool value = true;
};

/**
 * @brief Trait for checking if a type if a @p std::tuple containting only @p std::tuple
 */
template <typename T>
struct is_tuple_of_tuples {
  static constexpr bool value = false;
};
/// @overload
template <typename... T>
struct is_tuple_of_tuples<std::tuple<T...> > {
  static constexpr bool value = (is_tuple<T>::value && ...);
};

/////////////////////////////////////////////////

// apply operator+ elementwise to entries in two equally-sized tuples
template <typename... S, typename... T, int... I>
constexpr auto plus_helper(const std::tuple<S...>& A, const std::tuple<T...>& B, std::integer_sequence<int, I...>)
{
  return std::make_tuple((std::get<I>(A) + std::get<I>(B))...);
}

// apply operator- elementwise to entries in two equally-sized tuples
template <typename... S, typename... T, int... I>
constexpr auto minus_helper(const std::tuple<S...>& A, const std::tuple<T...>& B, std::integer_sequence<int, I...>)
{
  return std::make_tuple((std::get<I>(A) - std::get<I>(B))...);
}

// apply (entry * scale) to each entry in a tuple
template <typename... S, typename T, int... I>
constexpr auto mult_helper(const std::tuple<S...>& A, T scale, std::integer_sequence<int, I...>)
{
  return std::make_tuple(std::get<I>(A) * scale...);
}

// apply (scale * entry) to each entry in a tuple
template <typename S, typename... T, int... I>
constexpr auto mult_helper(S scale, const std::tuple<T...>& A, std::integer_sequence<int, I...>)
{
  return std::make_tuple((scale * std::get<I>(A))...);
}

// apply (entry / denominator) to each entry in a tuple
template <typename... S, typename T, int... I>
constexpr auto div_helper(const std::tuple<S...>& numerators, T denominator, std::integer_sequence<int, I...>)
{
  return std::make_tuple(std::get<I>(numerators) / denominator...);
}

// apply (numerator / entry) to each entry in a tuple
template <typename S, typename... T, int... I>
constexpr auto div_helper(S numerator, const std::tuple<T...>& denominators, std::integer_sequence<int, I...>)
{
  return std::make_tuple((numerator / std::get<I>(denominators))...);
}

// promote a double-precision value to a dual number representation that keeps track of
// derivatives w.r.t. more than 1 argument. It is assumed that 'arg' itself corresponds
// to the 'j'th argument. This function is not intended to be called outside of make_dual.
template <int j, int... i>
constexpr auto make_dual_helper(double arg, std::integer_sequence<int, i...>)
{
  using gradient_type = std::tuple<typename std::conditional<i == j, double, zero>::type...>;
  dual<gradient_type> arg_dual{};
  arg_dual.value                 = arg;
  std::get<j>(arg_dual.gradient) = 1.0;
  return arg_dual;
}

// promote a tensor of values to a dual tensor representation that keeps track of
// derivatives w.r.t. more than 1 argument. It is assumed that 'arg' itself corresponds
// to the 'j'th argument. This function is not intended to be called outside of make_dual.
template <int I, typename T, int... n, int... i>
constexpr auto make_dual_helper(const tensor<T, n...>& arg, std::integer_sequence<int, i...>)
{
  using gradient_type = std::tuple<typename std::conditional<i == I, tensor<T, n...>, zero>::type...>;
  tensor<dual<gradient_type>, n...> arg_dual{};
  for_constexpr<n...>([&](auto... j) {
    arg_dual(j...).value                       = arg(j...);
    std::get<I>(arg_dual(j...).gradient)(j...) = 1.0;
  });
  return arg_dual;
}

// promote a tuple of values to a tuple of dual value representations that keeps track of
// derivatives w.r.t. each tuple entry.
template <typename... T, int... i>
constexpr auto make_dual_helper(std::tuple<T...> args, std::integer_sequence<int, i...> seq)
{
  return std::make_tuple((make_dual_helper<i>(std::get<i>(args), seq))...);
}

// chain rule between a tuple of values and a single value,
// effectively equivalent to
//
// std::tuple df{
//   chain_rule(std::get<0>(df_dx), dx),
//   chain_rule(std::get<1>(df_dx), dx),
//   chain_rule(std::get<2>(df_dx), dx),
//   ...
// }
template <typename... T, typename S>
auto chain_rule_tuple_scale(std::tuple<T...> df_dx, S dx)
{
  return std::apply(
      [&](auto... each_component_of_df_dx) { return std::tuple{chain_rule(each_component_of_df_dx, dx)...}; }, df_dx);
}

// chain rule between two tuples of values, kind of like a "dot product"
// effectively equivalent to
//
// std::tuple df{
//   chain_rule(std::get<0>(df_dx), std::get<0>(dx)),
//   chain_rule(std::get<1>(df_dx), std::get<1>(dx)),
//   chain_rule(std::get<2>(df_dx), std::get<2>(dx)),
//   ...
// }
template <typename... T, typename... S, int... i>
auto chain_rule_tuple_vecvec(std::tuple<T...> df_dx, std::tuple<S...> dx, std::integer_sequence<int, i...>)
{
  return (chain_rule(std::get<i>(df_dx), std::get<i>(dx)) + ...);
}

// chain rule between a tuple-of-tuples, and another tuple, kind of like a "matrix vector product"
// effectively equivalent to
//
// clang-format off
// std::tuple df{
//   chain_rule(std::get<0>(std::get<0>(df_dx)), std::get<0>(dx)) + chain_rule(std::get<1>(std::get<0>(df_dx)), std::get<1>(dx)) + ... , 
//   chain_rule(std::get<0>(std::get<1>(df_dx)), std::get<0>(dx)) + chain_rule(std::get<1>(std::get<1>(df_dx)), std::get<1>(dx)) + ... , 
//   chain_rule(std::get<0>(std::get<2>(df_dx)), std::get<0>(dx)) + chain_rule(std::get<1>(std::get<2>(df_dx)), std::get<1>(dx)) + ... ,
//   ...
// }
// clang-format on
template <typename... T, typename... S>
auto chain_rule_tuple_matvec(std::tuple<T...> df_dx, std::tuple<S...> dx)
{
  auto int_seq = std::make_integer_sequence<int, int(sizeof...(S))>();

  return std::apply(
      [&](auto... each_component_of_df_dx) {
        return std::tuple{chain_rule_tuple_vecvec(each_component_of_df_dx, dx, int_seq)...};
      },
      df_dx);
}

}  // namespace detail

/**
 * @brief apply operator+ to each pair of entries in two similarly-sized tuples
 * @param[in] A a tuple of values
 * @param[in] B a tuple of values
 * Note: A and B must have the same size, and the type of the ith entry of the returned tuple
 * is given by decltype(std::get<i>(A) + std::get<i>(B))
 */
template <typename... S, typename... T>
constexpr auto operator+(const std::tuple<S...>& A, const std::tuple<T...>& B)
{
  static_assert(sizeof...(S) == sizeof...(T), "Error in operator+: tuple sizes must match");
  return detail::plus_helper(A, B, std::make_integer_sequence<int, int(sizeof...(S))>{});
}

/**
 * @brief apply operator+= to each pair of entries in two similarly-sized and similarly-typed tuples
 * @param[in] A a tuple of values
 * @param[in] B a tuple of values
 * Note: A and B must have the same size
 */
template <typename... S>
constexpr auto operator+=(std::tuple<S...>& A, const std::tuple<S...>& B)
{
  return A = A + B;
}

/**
 * @brief apply operator- to each pair of entries in two similarly-sized tuples
 * @param[in] A a tuple of values
 * @param[in] B a tuple of values
 * Note: A and B must have the same size, and the type of the ith entry of the returned tuple
 * is given by decltype(std::get<i>(A) - std::get<i>(B))
 */
template <typename... S, typename... T>
constexpr auto operator-(const std::tuple<S...>& A, const std::tuple<T...>& B)
{
  static_assert(sizeof...(S) == sizeof...(T), "Error in operator+: tuple sizes must match");
  return detail::minus_helper(A, B, std::make_integer_sequence<int, int(sizeof...(S))>{});
}

/**
 * @brief apply a scaling (from the right) to each entry in a tuple
 * @param[in] A a tuple of values
 * @param[in] scale the scaling factor
 * Note: the type of the ith entry of the returned tuple
 * is given by decltype(std::get<i>(A) * scale)
 */
template <typename... S, typename T>
constexpr auto operator*(const std::tuple<S...>& A, T scale)
{
  return detail::mult_helper(A, scale, std::make_integer_sequence<int, int(sizeof...(S))>{});
}

/**
 * @brief apply a scaling (from the left) to each entry in a tuple
 * @param[in] scale the scaling factor
 * @param[in] A a tuple of values
 * Note: the type of the ith entry of the returned tuple
 * is given by decltype(scale * std::get<i>(A))
 */
template <typename S, typename... T>
constexpr auto operator*(S scale, const std::tuple<T...>& A)
{
  return detail::mult_helper(scale, A, std::make_integer_sequence<int, int(sizeof...(T))>{});
}

/**
 * @brief divide each entry in a tuple by a denominator
 * @param[in] numerators a tuple of numerators
 * @param[in] denominator the denominator
 * Note: the type of the ith entry of the returned tuple
 * is given by decltype(std::get<i>(numerators) / denominator)
 */
template <typename... S, typename T>
constexpr auto operator/(const std::tuple<S...>& numerators, T denominator)
{
  return detail::div_helper(numerators, denominator, std::make_integer_sequence<int, int(sizeof...(S))>{});
}

/**
 * @brief return a tuple with entries defined by dividing a numerator value by each entry of a tuple
 * @param[in] numerator a numerator value
 * @param[in] denominators a tuple of denominator values
 * Note: the type of the ith entry of the returned tuple
 * is given by decltype(numerator / std::get<i>(denominators))
 */
template <typename S, typename... T>
constexpr auto operator/(S numerator, const std::tuple<T...>& denominators)
{
  return detail::div_helper(numerator, denominators, std::make_integer_sequence<int, int(sizeof...(T))>{});
}

/**
 * @brief Constructs a tuple of dual numbers from a parameter pack of values
 * @param[in] args The set of values
 * The gradients for each value will be set to 1 (or its tensor equivalent)
 */
template <typename... T>
constexpr auto make_dual(T... args)
{
  return detail::make_dual_helper(std::tuple{args...}, std::make_integer_sequence<int, int(sizeof...(T))>{});
}
/// @overload
template <typename... T>
constexpr auto make_dual(std::tuple<T...> args)
{
  return detail::make_dual_helper(args, std::make_integer_sequence<int, int(sizeof...(T))>{});
}

/**
 * @brief Retrieves the value components of a set of (possibly dual) numbers
 * @param[in] tuple_of_values The tuple of numbers to retrieve values from
 * @pre The tuple must contain only scalars or tensors of @p dual numbers or doubles
 */
template <typename... T>
auto get_value(std::tuple<T...> tuple_of_values)
{
  return std::apply([](auto... each_value) { return std::tuple{get_value(each_value)...}; }, tuple_of_values);
}

/**
 * @brief Retrieves the gradient components of a set of dual numbers
 * @param[in] arg The set of numbers to retrieve gradients from
 */
template <typename... T>
auto get_gradient(dual<std::tuple<T...> > arg)
{
  return std::apply([](auto... each_value) { return std::tuple{each_value...}; }, arg.gradient);
}
/// @overload
template <typename... T, int... n>
auto get_gradient(const tensor<dual<std::tuple<T...> >, n...>& arg)
{
  std::tuple<outer_product_t<tensor<double, n...>, T>...> g{};
  for_constexpr<n...>([&](auto... i) {
    for_constexpr<sizeof...(T)>([&](auto j) { std::get<j>(g)(i...) = std::get<j>(arg(i...).gradient); });
  });
  return g;
}
/// @overload
template <typename... T>
auto get_gradient(std::tuple<T...> tuple_of_values)
{
  return std::apply([](auto... each_value) { return std::tuple{get_gradient(each_value)...}; }, tuple_of_values);
}

/**
 * @brief entry point for combining derivatives from get_gradient(***) with the chain rule
 * calculates df = df_dx * dx for different possible combinations of tuples-of-tuples, tuples, tensors, and scalars
 * @param[in] df_dx the derivative of some transformation f
 * @param[in] dx a small change in the inputs to the transformation f
 */
template <typename S, typename T>
auto chain_rule(S df_dx, T dx)
{
  if constexpr ((detail::is_tuple_of_tuples<S>::value && detail::is_tuple<T>::value)) {
    return detail::chain_rule_tuple_matvec(df_dx, dx);
  } else if constexpr (detail::is_tuple<S>::value && detail::is_tuple<T>::value) {
    auto int_seq = std::make_integer_sequence<int, int(std::tuple_size<T>{})>();
    return detail::chain_rule_tuple_vecvec(df_dx, dx, int_seq);
  } else if constexpr (detail::is_tuple<S>::value && !detail::is_tuple<T>::value) {
    return detail::chain_rule_tuple_scale(df_dx, dx);
  } else {
    return chain_rule(df_dx, dx);
  }
}

}  // namespace serac

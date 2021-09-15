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
// serac::tuple< foo, bar > a;
// serac::tuple< baz, qux > b;
//
// serac::tuple sum1 = a + b;
// serac::tuple sum2{serac::get<0>(a) + serac::get<0>(b), serac::get<1>(a) + serac::get<1>(b)};

#pragma once

#include <utility>

#include "tuple.hpp"
#include "tensor.hpp"

namespace serac {

/// @cond
namespace detail {

/**
 * @brief Trait for checking if a type is a @p serac::tuple
 */
template <typename T>
struct is_tuple {
  // FIXME: Should we use std::false_type/std::true_type
  static constexpr bool value = false;
};
/// @overload
template <typename... T>
struct is_tuple<serac::tuple<T...> > {
  static constexpr bool value = true;
};

/**
 * @brief Trait for checking if a type if a @p serac::tuple containing only @p serac::tuple
 */
template <typename T>
struct is_tuple_of_tuples {
  static constexpr bool value = false;
};
/// @overload
template <typename... T>
struct is_tuple_of_tuples<serac::tuple<T...> > {
  static constexpr bool value = (is_tuple<T>::value && ...);
};

/////////////////////////////////////////////////

// promote a double-precision value to a dual number representation that keeps track of
// derivatives w.r.t. more than 1 argument. It is assumed that 'arg' itself corresponds
// to the 'j'th argument. This function is not intended to be called outside of make_dual.
template <int j, int... i>
constexpr auto make_dual_helper(double arg, std::integer_sequence<int, i...>)
{
  using gradient_type = serac::tuple<typename std::conditional<i == j, double, zero>::type...>;
  dual<gradient_type> arg_dual{};
  arg_dual.value                   = arg;
  serac::get<j>(arg_dual.gradient) = 1.0;
  return arg_dual;
}

// promote a tensor of values to a dual tensor representation that keeps track of
// derivatives w.r.t. more than 1 argument. It is assumed that 'arg' itself corresponds
// to the 'j'th argument. This function is not intended to be called outside of make_dual.
template <int I, typename T, int... n, int... i>
constexpr auto make_dual_helper(const tensor<T, n...>& arg, std::integer_sequence<int, i...>)
{
  using gradient_type = serac::tuple<typename std::conditional<i == I, tensor<T, n...>, zero>::type...>;
  tensor<dual<gradient_type>, n...> arg_dual{};
  for_constexpr<n...>([&](auto... j) {
    arg_dual(j...).value                         = arg(j...);
    serac::get<I>(arg_dual(j...).gradient)(j...) = 1.0;
  });
  return arg_dual;
}

// promote a tuple of values to a tuple of dual value representations that keeps track of
// derivatives w.r.t. each tuple entry.
template <typename... T, int... i>
constexpr auto make_dual_helper(serac::tuple<T...> args, std::integer_sequence<int, i...> seq)
{
  return serac::tuple{(make_dual_helper<i>(serac::get<i>(args), seq))...};
}

// chain rule between a tuple of values and a single value,
// effectively equivalent to
//
// serac::tuple df{
//   chain_rule(serac::get<0>(df_dx), dx),
//   chain_rule(serac::get<1>(df_dx), dx),
//   chain_rule(serac::get<2>(df_dx), dx),
//   ...
// }
template <typename... T, typename S>
SERAC_HOST_DEVICE auto chain_rule_tuple_scale(serac::tuple<T...> df_dx, S dx)
{
  return serac::apply(
      [&](auto... each_component_of_df_dx) { return serac::tuple{chain_rule(each_component_of_df_dx, dx)...}; }, df_dx);
}

// chain rule between two tuples of values, kind of like a "dot product"
// effectively equivalent to
//
// serac::tuple df{
//   chain_rule(serac::get<0>(df_dx), serac::get<0>(dx)),
//   chain_rule(serac::get<1>(df_dx), serac::get<1>(dx)),
//   chain_rule(serac::get<2>(df_dx), serac::get<2>(dx)),
//   ...
// }
template <typename... T, typename... S, int... i>
SERAC_HOST_DEVICE auto chain_rule_tuple_vecvec(serac::tuple<T...> df_dx, serac::tuple<S...> dx,
                                               std::integer_sequence<int, i...>)
{
  return (chain_rule(serac::get<i>(df_dx), serac::get<i>(dx)) + ...);
}

// chain rule between a tuple-of-tuples, and another tuple, kind of like a "matrix vector product"
// effectively equivalent to
//
// serac::tuple df{
//   chain_rule(serac::get<0>(serac::get<0>(df_dx)), serac::get<0>(dx)) +
//   chain_rule(serac::get<1>(serac::get<0>(df_dx)), serac::get<1>(dx)) + ... ,
//   chain_rule(serac::get<0>(serac::get<1>(df_dx)), serac::get<0>(dx)) +
//   chain_rule(serac::get<1>(serac::get<1>(df_dx)), serac::get<1>(dx)) + ... ,
//   chain_rule(serac::get<0>(serac::get<2>(df_dx)), serac::get<0>(dx)) +
//   chain_rule(serac::get<1>(serac::get<2>(df_dx)), serac::get<1>(dx)) + ... ,
//   ...
// }
template <typename... T, typename... S>
SERAC_HOST_DEVICE auto chain_rule_tuple_matvec(serac::tuple<T...> df_dx, serac::tuple<S...> dx)
{
  auto int_seq = std::make_integer_sequence<int, int(sizeof...(S))>();

  return serac::apply(
      [&](auto... each_component_of_df_dx) {
        return serac::tuple{chain_rule_tuple_vecvec(each_component_of_df_dx, dx, int_seq)...};
      },
      df_dx);
}

}  // namespace detail
/// @endcond

/**
 * @brief Constructs a tuple of dual numbers from a parameter pack of values
 * @param[in] args The set of values
 * The gradients for each value will be set to 1 (or its tensor equivalent)
 */
template <typename... T>
constexpr auto make_dual(T... args)
{
  return detail::make_dual_helper(serac::tuple{args...}, std::make_integer_sequence<int, int(sizeof...(T))>{});
}
/// @overload
template <typename... T>
constexpr auto make_dual(serac::tuple<T...> args)
{
  return detail::make_dual_helper(args, std::make_integer_sequence<int, int(sizeof...(T))>{});
}

/**
 * @brief Retrieves the value components of a set of (possibly dual) numbers
 * @param[in] tuple_of_values The tuple of numbers to retrieve values from
 * @pre The tuple must contain only scalars or tensors of @p dual numbers or doubles
 */
template <typename... T>
SERAC_HOST_DEVICE auto get_value(const serac::tuple<T...>& tuple_of_values)
{
  return serac::apply([](const auto&... each_value) { return serac::tuple{get_value(each_value)...}; },
                      tuple_of_values);
}

/**
 * @brief Retrieves the gradient components of a set of dual numbers
 * @param[in] arg The set of numbers to retrieve gradients from
 */
template <typename... T>
SERAC_HOST_DEVICE auto get_gradient(dual<serac::tuple<T...> > arg)
{
  return serac::apply([](auto... each_value) { return serac::tuple{each_value...}; }, arg.gradient);
}

/// @overload
template <typename... T, int... n>
SERAC_HOST_DEVICE auto get_gradient(const tensor<dual<serac::tuple<T...> >, n...>& arg)
{
  serac::tuple<outer_product_t<tensor<double, n...>, T>...> g{};
  for_constexpr<n...>([&](auto... i) {
    for_constexpr<sizeof...(T)>([&](auto j) { serac::get<j>(g)(i...) = serac::get<j>(arg(i...).gradient); });
  });
  return g;
}
/// @overload
template <typename... T>
SERAC_HOST_DEVICE auto get_gradient(serac::tuple<T...> tuple_of_values)
{
  return serac::apply([](auto... each_value) { return serac::tuple{get_gradient(each_value)...}; }, tuple_of_values);
}

/**
 * @brief entry point for combining derivatives from get_gradient(***) with the chain rule
 * calculates df = df_dx * dx for different possible combinations of tuples-of-tuples, tuples, tensors, and scalars
 * @param[in] df_dx the derivative of some transformation f
 * @param[in] dx a small change in the inputs to the transformation f
 * 
 * @note the weird implementation of these conditional statements is a work-around for a compiler warning w/ nvcc
 */
template <typename S, typename T>
SERAC_HOST_DEVICE auto chain_rule(S df_dx, T dx)
{
  constexpr bool matvec = (detail::is_tuple_of_tuples<S>::value && detail::is_tuple<T>::value);
  if constexpr (matvec) {
    return detail::chain_rule_tuple_matvec(df_dx, dx);
  } 

  constexpr bool vecvec = !matvec && (detail::is_tuple<S>::value && detail::is_tuple<T>::value);
  if constexpr (vecvec) {
    auto int_seq = std::make_integer_sequence<int, int(serac::tuple_size<T>{})>();
    return detail::chain_rule_tuple_vecvec(df_dx, dx, int_seq);
  } 
  
  constexpr bool vecscalar = !vecvec && (detail::is_tuple<S>::value && !detail::is_tuple<T>::value);
  if constexpr (vecscalar) {
    return detail::chain_rule_tuple_scale(df_dx, dx);
  } 

  constexpr bool scalarscalar = !vecscalar;
  if constexpr (scalarscalar) {
    return chain_rule(df_dx, dx);
  }
}

}  // namespace serac

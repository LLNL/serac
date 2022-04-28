// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
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
// camp::tuple< foo, bar > a;
// camp::tuple< baz, qux > b;
//
// camp::tuple sum1 = a + b;
// camp::tuple sum2{camp::get<0>(a) + camp::get<0>(b), camp::get<1>(a) + camp::get<1>(b)};

#pragma once

#include <utility>

#include "tuple.hpp"
#include "tensor.hpp"

namespace serac {

/// @cond
namespace detail {

/**
 * @brief Trait for checking if a type is a @p camp::tuple
 */
template <typename T>
struct is_tuple : std::false_type {
};

/// @overload
template <typename... T>
struct is_tuple<camp::tuple<T...> > : std::true_type {
};

/**
 * @brief Trait for checking if a type if a @p camp::tuple containing only @p camp::tuple
 */
template <typename T>
struct is_tuple_of_tuples : std::false_type {
};

/// @overload
template <typename... T>
struct is_tuple_of_tuples<camp::tuple<T...> > {
  static constexpr bool value = (is_tuple<T>::value && ...);
};

/////////////////////////////////////////////////

// promote a double-precision value to a dual number representation that keeps track of
// derivatives w.r.t. more than 1 argument. It is assumed that 'arg' itself corresponds
// to the 'j'th argument. This function is not intended to be called outside of make_dual.
template <int j, int... i>
constexpr auto make_dual_helper(double arg, std::integer_sequence<int, i...>)
{
  using gradient_type = camp::tuple<typename std::conditional<i == j, double, zero>::type...>;
  dual<gradient_type> arg_dual{};
  arg_dual.value                   = arg;
  camp::get<j>(arg_dual.gradient) = 1.0;
  return arg_dual;
}

// promote a tensor of values to a dual tensor representation that keeps track of
// derivatives w.r.t. more than 1 argument. It is assumed that 'arg' itself corresponds
// to the 'j'th argument. This function is not intended to be called outside of make_dual.
template <int I, typename T, int... n, int... i>
constexpr auto make_dual_helper(const tensor<T, n...>& arg, std::integer_sequence<int, i...>)
{
  using gradient_type = camp::tuple<typename std::conditional<i == I, tensor<T, n...>, zero>::type...>;
  tensor<dual<gradient_type>, n...> arg_dual{};
  for_constexpr<n...>([&](auto... j) {
    arg_dual(j...).value                         = arg(j...);
    camp::get<I>(arg_dual(j...).gradient)(j...) = 1.0;
  });
  return arg_dual;
}

// promote a tuple of values to a tuple of dual value representations that keeps track of
// derivatives w.r.t. each tuple entry.
template <typename... T, int... i>
constexpr auto make_dual_helper(camp::tuple<T...> args, std::integer_sequence<int, i...> seq)
{
  return camp::tuple{(make_dual_helper<i>(camp::get<i>(args), seq))...};
}

}  // namespace detail

template <int i, typename S, typename T>
struct one_hot_helper;

template <int i, int... I, typename T>
struct one_hot_helper<i, std::integer_sequence<int, I...>, T> {
  using type = camp::tuple<std::conditional_t<i == I, T, zero>...>;
};

template <int i, int n, typename T>
struct one_hot : public one_hot_helper<i, std::make_integer_sequence<int, n>, T> {
};
/// @endcond

/**
 * @brief a tuple type with n entries, all of which are of type `serac::zero`,
 * except for the i^{th} entry, which is of type T
 *
 *  e.g. one_hot_t< 2, 4, T > == tuple<zero, zero, T, zero>
 */
template <int i, int n, typename T>
using one_hot_t = typename one_hot<i, n, T>::type;

/// @overload
template <int i, int N>
SERAC_HOST_DEVICE constexpr auto make_dual_helper(zero)
{
  return zero{};
}

/**
 * @tparam i the index where the non-`serac::zero` derivative term appears
 * @tparam N how many entries in the gradient type
 *
 * @brief promote a double value to dual number with a one_hot_t< i, N, double > gradient type
 * @param arg the value to be promoted
 */
template <int i, int N>
SERAC_HOST_DEVICE constexpr auto make_dual_helper(double arg)
{
  using gradient_t = one_hot_t<i, N, double>;
  dual<gradient_t> arg_dual{};
  arg_dual.value                   = arg;
  camp::get<i>(arg_dual.gradient) = 1.0;
  return arg_dual;
}

/**
 * @tparam i the index where the non-`serac::zero` derivative term appears
 * @tparam N how many entries in the gradient type
 *
 * @brief promote a tensor value to dual number with a one_hot_t< i, N, tensor > gradient type
 * @param arg the value to be promoted
 */
template <int i, int N, typename T, int... n>
SERAC_HOST_DEVICE constexpr auto make_dual_helper(const tensor<T, n...>& arg)
{
  using gradient_t = one_hot_t<i, N, tensor<T, n...> >;
  tensor<dual<gradient_t>, n...> arg_dual{};
  for_constexpr<n...>([&](auto... j) {
    arg_dual(j...).value                         = arg(j...);
    camp::get<i>(arg_dual(j...).gradient)(j...) = 1.0;
  });
  return arg_dual;
}

/**
 * @tparam T0 the first type of the tuple argument
 * @tparam T1 the first type of the tuple argument
 *
 * @brief Promote a tuple of values to their corresponding dual types
 * @param args the values to be promoted
 *
 * example:
 * @code{.cpp}
 * camp::tuple < double, tensor< double, 3 > > f{};
 *
 * camp::tuple <
 *   dual < camp::tuple < double, zero > >
 *   tensor < dual < camp::tuple < zero, tensor< double, 3 > >, 3 >
 * > dual_of_f = make_dual(f);
 * @endcode
 */
template <typename T0, typename T1>
SERAC_HOST_DEVICE constexpr auto make_dual(const camp::tuple<T0, T1>& args)
{
  return camp::tuple{make_dual_helper<0, 2>(get<0>(args)), make_dual_helper<1, 2>(get<1>(args))};
}

/**
 * @tparam dualify specify whether or not the value should be made into its dual type
 * @tparam T the type of the value passed in
 *
 * @brief a function that optionally (decided at compile time) converts a value to its dual type
 * @param x the values to be promoted
 */
template <bool dualify, typename T>
SERAC_HOST_DEVICE auto promote_to_dual_when(const T& x)
{
  if constexpr (dualify) {
    return make_dual(x);
  }
  if constexpr (!dualify) {
    return x;
  }
}

/// @brief layer of indirection required to implement `make_dual_wrt`
template <int n, typename... T, int... i>
SERAC_HOST_DEVICE constexpr auto make_dual_helper(const camp::tuple<T...>& args, std::integer_sequence<int, i...>)
{
  // Sam: it took me longer than I'd like to admit to find this issue, so here's an explanation
  //
  // note: we use camp::make_tuple(...) instead of camp::tuple{...} here because if
  // the first argument passed in is of type `camp::tuple < camp::tuple < T ... > >`
  // then doing something like
  //
  // camp::tuple{camp::get<i>(args)...};
  //
  // will be expand to something like
  //
  // camp::tuple{camp::tuple< T ... >{}};
  //
  // which invokes the copy ctor, returning a `camp::tuple< T ... >`
  // instead of `camp::tuple< camp::tuple < T ... > >`
  //
  // but camp::make_tuple(camp::get<i>(args)...) will never accidentally trigger the copy ctor
  return camp::make_tuple(promote_to_dual_when<i == n>(camp::get<i>(args))...);
}

/**
 * @tparam n the index of the tuple argument to be made into a dual number
 * @tparam T the types of the values in the tuple
 *
 * @brief take a tuple of values, and promote the `n`th one to a one-hot dual number of the appropriate type
 * @param args the values to be promoted
 */
template <int n, typename... T>
constexpr auto make_dual_wrt(const camp::tuple<T...>& args)
{
  return make_dual_helper<n>(args, std::make_integer_sequence<int, int(sizeof...(T))>{});
}

/**
 * @brief Retrieves the value components of a set of (possibly dual) numbers
 * @param[in] tuple_of_values The tuple of numbers to retrieve values from
 * @pre The tuple must contain only scalars or tensors of @p dual numbers or doubles
 */
template <typename... T>
SERAC_HOST_DEVICE auto get_value(const camp::tuple<T...>& tuple_of_values)
{
  return serac::apply([](const auto&... each_value) { return camp::tuple{get_value(each_value)...}; },
                      tuple_of_values);
}

/**
 * @brief Retrieves the gradient components of a set of dual numbers
 * @param[in] arg The set of numbers to retrieve gradients from
 */
template <typename... T>
SERAC_HOST_DEVICE auto get_gradient(dual<camp::tuple<T...> > arg)
{
  return serac::apply([](auto... each_value) { return camp::tuple{each_value...}; }, arg.gradient);
}

/// @overload
template <typename... T, int... n>
SERAC_HOST_DEVICE auto get_gradient(const tensor<dual<camp::tuple<T...> >, n...>& arg)
{
  camp::tuple<outer_product_t<tensor<double, n...>, T>...> g{};
  for_constexpr<n...>([&](auto... i) {
    for_constexpr<sizeof...(T)>([&](auto j) { camp::get<j>(g)(i...) = camp::get<j>(arg(i...).gradient); });
  });
  return g;
}

/// @overload
template <typename... T>
SERAC_HOST_DEVICE auto get_gradient(camp::tuple<T...> tuple_of_values)
{
  return serac::apply([](auto... each_value) { return camp::tuple{get_gradient(each_value)...}; }, tuple_of_values);
}

}  // namespace serac

// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file metaprogramming.hpp
 *
 * @brief Utilities for C++ metaprogramming
 */

#pragma once

#include <tuple>
#include <utility>
#include <type_traits>

#include "serac/infrastructure/accelerator.hpp"

/**
 * @brief return the Ith integer in `{n...}`
 */
template <int I, int... n>
constexpr auto get(std::integer_sequence<int, n...>)
{
  constexpr int values[sizeof...(n)] = {n...};
  return values[I];
}

/// @cond
namespace detail {

template <typename T>
struct always_false : std::false_type {
};

/**
 * @brief unfortunately std::integral_constant doesn't have __host__ __device__ annotations
 * and we're not using --expt-relaxed-constexpr, so we need to implement something similar
 * to use it in a device context
 *
 * @tparam i the value represented by this struct
 */
template <int i>
struct integral_constant {
  SERAC_HOST_DEVICE constexpr operator int() { return i; }
};

SERAC_SUPPRESS_NVCC_HOSTDEVICE_WARNING
template <typename lambda, int... i>
SERAC_HOST_DEVICE constexpr void for_constexpr(lambda&& f, integral_constant<i>... args)
{
  f(args...);
}

SERAC_SUPPRESS_NVCC_HOSTDEVICE_WARNING
template <int... n, typename lambda, typename... arg_types>
SERAC_HOST_DEVICE constexpr void for_constexpr(lambda&& f, std::integer_sequence<int, n...>, arg_types... args)
{
  (detail::for_constexpr(f, args..., integral_constant<n>{}), ...);
}

template <typename T, int N, int... I>
constexpr std::array<T, N + 1> append_helper(std::array<T, N> a, T t, std::index_sequence<I...>)
{
  return std::array<T, N + 1>{a[I]..., t};
}

template <typename T, int N, int... I>
constexpr std::array<T, N + 1> prepend_helper(T&& t, std::array<T, N>&& a, std::index_sequence<I...>)
{
  return std::array<T, N + 1>{t, a[I]...};
}

}  // namespace detail
/// @endcond

/**
 * @brief multidimensional loop tool that evaluates the lambda body inside the innermost loop.
 *
 * @tparam n integer template arguments describing the shape of the iteration space
 * @tparam lambda the type of the functor object to be executed in the loop
 *
 * @note
 * \code{.cpp}
 * for_constexpr< 2, 3 >([](auto i, auto j) { std::cout << i << " " << j << std::endl; }
 * \endcode
 * will print:
 * \code{.cpp}
 * 0 0
 * 0 1
 * 0 2
 * 1 0
 * 1 1
 * 1 2
 * \endcode
 *
 * @note latter integer template parameters correspond to more nested loops
 *
 * @note The lambda function should be a callable object taking sizeof ... (n) arguments.
 * Anything returned from f() will be discarded
 * note: this forces multidimensional loop unrolling, which can be beneficial for
 * runtime performance, but can hurt compile time and executable size as the loop
 * dimensions become larger.
 *
 */
template <int... n, typename lambda>
SERAC_HOST_DEVICE constexpr void for_constexpr(lambda&& f)
{
  detail::for_constexpr(f, std::make_integer_sequence<int, n>{}...);
}

template <typename T, int N>
constexpr std::array<T, N + 1> append(std::array<T, N> a, T t)
{
  return detail::append_helper(a, t, std::make_index_sequence<N>());
}

template <typename T, int N>
constexpr std::array<T, N + 1> prepend(T&& t, std::array<T, N>&& a)
{
  return detail::prepend_helper(t, a, std::make_index_sequence<N>());
}
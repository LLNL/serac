// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
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
template <typename lambda, int... i>
inline constexpr void for_constexpr(lambda&& f, std::integral_constant<int, i>... args)
{
  f(args...);
}

template <int... n, typename lambda, typename... arg_types>
inline constexpr void for_constexpr(lambda&& f, std::integer_sequence<int, n...>, arg_types... args)
{
  (detail::for_constexpr(f, args..., std::integral_constant<int, n>{}), ...);
}

template <int r, int... n, int... i>
constexpr auto remove_helper(std::integer_sequence<int, n...>, std::integer_sequence<int, i...>)
{
  return std::integer_sequence<int, get<i + (i >= r)>(std::integer_sequence<int, n...>{})...>{};
}
}  // namespace detail
/// @endcond

/**
 * @brief return the `first` argument in a variadic list
 */
template <typename... T>
constexpr auto first(T... args)
{
  return std::get<0>(std::tuple{args...});
}

/**
 * @brief return the `last` argument in a variadic list
 */
template <typename... T>
constexpr auto last(T... args)
{
  return std::get<sizeof...(T) - 1>(std::tuple{args...});
}

/**
 * @brief return a new std::integer_sequence, after removing the rth entry from another std::integer_sequence
 */
template <int r, int... n>
constexpr auto remove(std::integer_sequence<int, n...> seq)
{
  return detail::remove_helper<r>(seq, std::make_integer_sequence<int, int(sizeof...(n)) - 1>{});
}

/**
 * @brief return the concatenation of two std::integer_sequences
 */
template <int... n1, int... n2>
constexpr auto join(std::integer_sequence<int, n1...>, std::integer_sequence<int, n2...>)
{
  return std::integer_sequence<int, n1..., n2...>{};
}

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
inline constexpr void for_constexpr(lambda&& f)
{
  detail::for_constexpr(f, std::make_integer_sequence<int, n>{}...);
}

/**
 * @brief overload
 */
template <int n1, typename lambda>
void for_loop(lambda f)
{
  for (int i = 0; i < n1; i++) {
    f(i);
  }
}

/**
 * @brief overload
 */
template <int n1, int n2, typename lambda>
void for_loop(lambda f)
{
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      f(i, j);
    }
  }
}

/**
 * @brief overload
 */
template <int n1, int n2, int n3, typename lambda>
void for_loop(lambda f)
{
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      for (int k = 0; k < n3; k++) {
        f(i, j, k);
      }
    }
  }
}

/**
 * @brief overload
 */
template <int n1, int n2, int n3, int n4, typename lambda>
void for_loop(lambda f)
{
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      for (int k = 0; k < n3; k++) {
        for (int l = 0; l < n4; l++) {
          f(i, j, k, l);
        }
      }
    }
  }
}

/**
 * @brief overload
 */
template <int n1, int n2, int n3, int n4, int n5, typename lambda>
void for_loop(lambda f)
{
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      for (int k = 0; k < n3; k++) {
        for (int l = 0; l < n4; l++) {
          for (int m = 0; m < n5; m++) {
            f(i, j, k, l, m);
          }
        }
      }
    }
  }
}

/**
 * @brief overload
 */
template <int n1, int n2, int n3, int n4, int n5, int n6, typename lambda>
void for_loop(lambda f)
{
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      for (int k = 0; k < n3; k++) {
        for (int l = 0; l < n4; l++) {
          for (int m = 0; m < n5; m++) {
            for (int n = 0; n < n6; n++) {
              f(i, j, k, l, m, n);
            }
          }
        }
      }
    }
  }
}

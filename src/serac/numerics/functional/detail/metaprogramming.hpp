// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
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

/**
 * @overload
 */
template <int n1, typename lambda>
void for_loop(lambda f)
{
  for (int i = 0; i < n1; i++) {
    f(i);
  }
}

/**
 * @overload
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
 * @overload
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
 * @overload
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
 * @overload
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
 * @overload
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

// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file tuple.hpp
 *
 * @brief Implements a std::tuple-like object that works in CUDA kernels
 */
#pragma once

#include <utility>

#include "serac/infrastructure/accelerator.hpp"

namespace serac {

/**
 * @tparam T the types stored in the tuple
 * @brief This is a class that mimics most of std::tuple's interface,
 * except that it is usable in CUDA kernels and admits some arithmetic operator overloads.
 *
 * see https://en.cppreference.com/w/cpp/utility/tuple for more information about std::tuple
 */
template <typename... T>
struct tuple {
};

template <typename T0>
struct tuple<T0> {
  T0 v0;
};

template <typename T0, typename T1>
struct tuple<T0, T1> {
  T0 v0;
  T1 v1;
};

template <typename T0, typename T1, typename T2>
struct tuple<T0, T1, T2> {
  T0 v0;
  T1 v1;
  T2 v2;
};

template <typename T0, typename T1, typename T2, typename T3>
struct tuple<T0, T1, T2, T3> {
  T0 v0;
  T1 v1;
  T2 v2;
  T3 v3;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4>
struct tuple<T0, T1, T2, T3, T4> {
  T0 v0;
  T1 v1;
  T2 v2;
  T3 v3;
  T4 v4;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
struct tuple<T0, T1, T2, T3, T4, T5> {
  T0 v0;
  T1 v1;
  T2 v2;
  T3 v3;
  T4 v4;
  T5 v5;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
struct tuple<T0, T1, T2, T3, T4, T5, T6> {
  T0 v0;
  T1 v1;
  T2 v2;
  T3 v3;
  T4 v4;
  T5 v5;
  T6 v6;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
struct tuple<T0, T1, T2, T3, T4, T5, T6, T7> {
  T0 v0;
  T1 v1;
  T2 v2;
  T3 v3;
  T4 v4;
  T5 v5;
  T6 v6;
  T7 v7;
};

template <typename... T>
tuple(T...) -> tuple<T...>;

template <class... Types>
struct tuple_size {
};

template <class... Types>
struct tuple_size<serac::tuple<Types...> > : std::integral_constant<std::size_t, sizeof...(Types)> {
};

/**
 * @tparam i the tuple index to access
 * @tparam T the types stored in the tuple
 * @brief return a reference to the ith tuple entry
 */
template <int i, typename... T>
SERAC_HOST_DEVICE constexpr auto& get(tuple<T...>& values)
{
  static_assert(i < sizeof...(T), "");
  if constexpr (i == 0) {
    return values.v0;
  }
  if constexpr (i == 1) {
    return values.v1;
  }
  if constexpr (i == 2) {
    return values.v2;
  }
  if constexpr (i == 3) {
    return values.v3;
  }
  if constexpr (i == 4) {
    return values.v4;
  }
  if constexpr (i == 5) {
    return values.v5;
  }
  if constexpr (i == 6) {
    return values.v6;
  }
  if constexpr (i == 7) {
    return values.v7;
  }
}

/**
 * @tparam i the tuple index to access
 * @tparam T the types stored in the tuple
 * @brief return a copy of the ith tuple entry
 */
template <int i, typename... T>
SERAC_HOST_DEVICE constexpr auto get(const tuple<T...>& values)
{
  static_assert(i < sizeof...(T), "");
  if constexpr (i == 0) {
    return values.v0;
  }
  if constexpr (i == 1) {
    return values.v1;
  }
  if constexpr (i == 2) {
    return values.v2;
  }
  if constexpr (i == 3) {
    return values.v3;
  }
  if constexpr (i == 4) {
    return values.v4;
  }
  if constexpr (i == 5) {
    return values.v5;
  }
  if constexpr (i == 6) {
    return values.v6;
  }
  if constexpr (i == 7) {
    return values.v7;
  }
}

template <typename... S, typename... T, int... i>
SERAC_HOST_DEVICE constexpr auto plus_helper(const tuple<S...>& x, const tuple<T...>& y,
                                             std::integer_sequence<int, i...>)
{
  return tuple{get<i>(x) + get<i>(y)...};
}

/**
 * @tparam S the types stored in the tuple x
 * @tparam T the types stored in the tuple y
 * @param x a tuple of values
 * @param y a tuple of values
 * @brief return a tuple of values defined by elementwise sum of x and y
 */
template <typename... S, typename... T>
SERAC_HOST_DEVICE constexpr auto operator+(const tuple<S...>& x, const tuple<T...>& y)
{
  static_assert(sizeof...(S) == sizeof...(T));
  return plus_helper(x, y, std::make_integer_sequence<int, static_cast<int>(sizeof...(S))>());
}

template <typename... S, typename... T, int... i>
SERAC_HOST_DEVICE constexpr auto minus_helper(const tuple<S...>& x, const tuple<T...>& y,
                                              std::integer_sequence<int, i...>)
{
  return tuple{get<i>(x) - get<i>(y)...};
}

/**
 * @tparam S the types stored in the tuple x
 * @tparam T the types stored in the tuple y
 * @param x a tuple of values
 * @param y a tuple of values
 * @brief return a tuple of values defined by elementwise difference of x and y
 */
template <typename... S, typename... T>
SERAC_HOST_DEVICE constexpr auto operator-(const tuple<S...>& x, const tuple<T...>& y)
{
  static_assert(sizeof...(S) == sizeof...(T));
  return minus_helper(x, y, std::make_integer_sequence<int, static_cast<int>(sizeof...(S))>());
}

template <typename... S, typename... T, int... i>
SERAC_HOST_DEVICE constexpr auto div_helper(const tuple<S...>& x, const tuple<T...>& y,
                                            std::integer_sequence<int, i...>)
{
  return tuple{get<i>(x) / get<i>(y)...};
}

/**
 * @tparam S the types stored in the tuple x
 * @tparam T the types stored in the tuple y
 * @param x a tuple of values
 * @param y a tuple of values
 * @brief return a tuple of values defined by elementwise division of x by y
 */
template <typename... S, typename... T>
SERAC_HOST_DEVICE constexpr auto operator/(const tuple<S...>& x, const tuple<T...>& y)
{
  static_assert(sizeof...(S) == sizeof...(T));
  return div_helper(x, y, std::make_integer_sequence<int, static_cast<int>(sizeof...(S))>());
}

template <typename... T, int... i>
SERAC_HOST_DEVICE constexpr auto div_helper(const double a, const tuple<T...>& x, std::integer_sequence<int, i...>)
{
  return tuple{a / get<i>(x)...};
}

template <typename... T, int... i>
SERAC_HOST_DEVICE constexpr auto div_helper(const tuple<T...>& x, const double a, std::integer_sequence<int, i...>)
{
  return tuple{get<i>(x) / a...};
}

/**
 * @tparam T the types stored in the tuple x
 * @param a the numerator
 * @param y a tuple of denominator values
 * @brief return a tuple of values defined by division of a by the elements of x
 */
template <typename... T>
SERAC_HOST_DEVICE constexpr auto operator/(const double a, const tuple<T...>& x)
{
  return div_helper(a, x, std::make_integer_sequence<int, static_cast<int>(sizeof...(T))>());
}

/**
 * @tparam T the types stored in the tuple y
 * @param x a tuple of numerator values
 * @param a a denominator
 * @brief return a tuple of values defined by elementwise division of x by a
 */
template <typename... T>
SERAC_HOST_DEVICE constexpr auto operator/(const tuple<T...>& x, const double a)
{
  return div_helper(x, a, std::make_integer_sequence<int, static_cast<int>(sizeof...(T))>());
}

template <typename... S, typename... T, int... i>
SERAC_HOST_DEVICE constexpr auto mult_helper(const tuple<S...>& x, const tuple<T...>& y,
                                             std::integer_sequence<int, i...>)
{
  return tuple{get<i>(x) * get<i>(y)...};
}

/**
 * @tparam S the types stored in the tuple x
 * @tparam T the types stored in the tuple y
 * @param x a tuple of values
 * @param y a tuple of values
 * @brief return a tuple of values defined by elementwise multiplication of x and y
 */
template <typename... S, typename... T>
SERAC_HOST_DEVICE constexpr auto operator*(const tuple<S...>& x, const tuple<T...>& y)
{
  static_assert(sizeof...(S) == sizeof...(T));
  return mult_helper(x, y, std::make_integer_sequence<int, static_cast<int>(sizeof...(S))>());
}

template <typename... T, int... i>
SERAC_HOST_DEVICE constexpr auto mult_helper(const double a, const tuple<T...>& x, std::integer_sequence<int, i...>)
{
  return tuple{a * get<i>(x)...};
}

template <typename... T, int... i>
SERAC_HOST_DEVICE constexpr auto mult_helper(const tuple<T...>& x, const double a, std::integer_sequence<int, i...>)
{
  return tuple{get<i>(x) * a...};
}

/**
 * @tparam T the types stored in the tuple
 * @param a a scaling factor
 * @param x the tuple object
 * @brief multiply each component of x by the value a on the left
 */
template <typename... T>
SERAC_HOST_DEVICE constexpr auto operator*(const double a, const tuple<T...>& x)
{
  return mult_helper(a, x, std::make_integer_sequence<int, static_cast<int>(sizeof...(T))>());
}

/**
 * @tparam T the types stored in the tuple
 * @param x the tuple object
 * @param a a scaling factor
 * @brief multiply each component of x by the value a on the right
 */
template <typename... T>
SERAC_HOST_DEVICE constexpr auto operator*(const tuple<T...>& x, const double a)
{
  return mult_helper(x, a, std::make_integer_sequence<int, static_cast<int>(sizeof...(T))>());
}

template <typename lambda, typename... T, int... i>
SERAC_HOST_DEVICE auto apply_helper(lambda f, tuple<T...>& args, std::integer_sequence<int, i...>)
{
  return f(get<i>(args)...);
}

/**
 * @tparam lambda a callable type
 * @tparam T the types of arguments to be passed in to f
 * @param f the callable object
 * @param args a tuple of arguments
 * @brief a way of passing an n-tuple to a function that expects n separate arguments
 *
 *   e.g. foo(bar, baz) is equivalent to apply(foo, serac::tuple(bar,baz));
 */
template <typename lambda, typename... T>
SERAC_HOST_DEVICE auto apply(lambda f, tuple<T...>& args)
{
  return apply_helper(f, std::move(args), std::make_integer_sequence<int, static_cast<int>(sizeof...(T))>());
}

template <typename lambda, typename... T, int... i>
SERAC_HOST_DEVICE auto apply_helper(lambda f, const tuple<T...>& args, std::integer_sequence<int, i...>)
{
  return f(get<i>(args)...);
}

/**
 * @tparam lambda a callable type
 * @tparam T the types of arguments to be passed in to f
 * @param f the callable object
 * @param args a tuple of arguments
 * @brief a way of passing an n-tuple to a function that expects n separate arguments
 *
 *   e.g. foo(bar, baz) is equivalent to apply(foo, serac::tuple(bar,baz));
 */
template <typename lambda, typename... T>
SERAC_HOST_DEVICE auto apply(lambda f, const tuple<T...>& args)
{
  return apply_helper(f, std::move(args), std::make_integer_sequence<int, static_cast<int>(sizeof...(T))>());
}

}  // namespace serac

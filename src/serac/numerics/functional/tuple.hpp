// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
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

#include "camp/tuple.hpp"

#include "serac/infrastructure/accelerator.hpp"

namespace serac {

/**
 * @brief A helper function for the + operator of tuples
 *
 * @tparam S the types stored in the tuple x
 * @tparam T the types stored in the tuple y
 * @tparam i The integer sequence to i
 * @param x tuple of values
 * @param y tuple of values
 * @return the returned tuple sum
 */
template <typename... S, typename... T, int... i>
SERAC_HOST_DEVICE constexpr auto plus_helper(const camp::tuple<S...>& x, const camp::tuple<T...>& y,
                                             std::integer_sequence<int, i...>)
{
  return camp::tuple{camp::get<i>(x) + camp::get<i>(y)...};
}

/**
 * @tparam S the types stored in the tuple x
 * @tparam T the types stored in the tuple y
 * @param x a tuple of values
 * @param y a tuple of values
 * @brief return a tuple of values defined by elementwise sum of x and y
 */
template <typename... S, typename... T>
SERAC_HOST_DEVICE constexpr auto operator+(const camp::tuple<S...>& x, const camp::tuple<T...>& y)
{
  static_assert(sizeof...(S) == sizeof...(T));
  return plus_helper(x, y, std::make_integer_sequence<int, static_cast<int>(sizeof...(S))>());
}

/**
 * @brief A helper function for the += operator of tuples
 *
 * @tparam T the types stored in the tuples x and y
 * @tparam i integer sequence used to index the tuples
 * @param x tuple of values to be incremented
 * @param y tuple of increment values
 */
template <typename... T, int... i>
SERAC_HOST_DEVICE constexpr void plus_equals_helper(camp::tuple<T...>& x, const camp::tuple<T...>& y,
                                                    std::integer_sequence<int, i...>)
{
  ((camp::get<i>(x) += camp::get<i>(y)), ...);
}

/**
 * @tparam T the types stored in the tuples x and y
 * @param x a tuple of values
 * @param y a tuple of values
 * @brief add values contained in y, to the tuple x
 */
template <typename... T>
SERAC_HOST_DEVICE constexpr auto operator+=(camp::tuple<T...>& x, const camp::tuple<T...>& y)
{
  return plus_equals_helper(x, y, std::make_integer_sequence<int, static_cast<int>(sizeof...(T))>());
}

/**
 * @brief A helper function for the -= operator of tuples
 *
 * @tparam T the types stored in the tuples x and y
 * @tparam i integer sequence used to index the tuples
 * @param x tuple of values to be subracted from
 * @param y tuple of values to subtract from x
 */
template <typename... T, int... i>
SERAC_HOST_DEVICE constexpr void minus_equals_helper(camp::tuple<T...>& x, const camp::tuple<T...>& y,
                                                     std::integer_sequence<int, i...>)
{
  ((camp::get<i>(x) -= camp::get<i>(y)), ...);
}

/**
 * @tparam T the types stored in the tuples x and y
 * @param x a tuple of values
 * @param y a tuple of values
 * @brief add values contained in y, to the tuple x
 */
template <typename... T>
SERAC_HOST_DEVICE constexpr auto operator-=(camp::tuple<T...>& x, const camp::tuple<T...>& y)
{
  return minus_equals_helper(x, y, std::make_integer_sequence<int, static_cast<int>(sizeof...(T))>());
}

/**
 * @brief A helper function for the - operator of tuples
 *
 * @tparam S the types stored in the tuple x
 * @tparam T the types stored in the tuple y
 * @tparam i The integer sequence to i
 * @param x tuple of values
 * @param y tuple of values
 * @return the returned tuple difference
 */
template <typename... S, typename... T, int... i>
SERAC_HOST_DEVICE constexpr auto minus_helper(const camp::tuple<S...>& x, const camp::tuple<T...>& y,
                                              std::integer_sequence<int, i...>)
{
  return camp::tuple{camp::get<i>(x) - camp::get<i>(y)...};
}

/**
 * @tparam S the types stored in the tuple x
 * @tparam T the types stored in the tuple y
 * @param x a tuple of values
 * @param y a tuple of values
 * @brief return a tuple of values defined by elementwise difference of x and y
 */
template <typename... S, typename... T>
SERAC_HOST_DEVICE constexpr auto operator-(const camp::tuple<S...>& x, const camp::tuple<T...>& y)
{
  static_assert(sizeof...(S) == sizeof...(T));
  return minus_helper(x, y, std::make_integer_sequence<int, static_cast<int>(sizeof...(S))>());
}

/**
 * @brief A helper function for the - operator of tuples
 *
 * @tparam T the types stored in the tuple y
 * @tparam i The integer sequence to i
 * @param x tuple of values
 * @return the returned tuple difference
 */
template <typename... T, int... i>
SERAC_HOST_DEVICE constexpr auto unary_minus_helper(const camp::tuple<T...>& x, std::integer_sequence<int, i...>)
{
  return camp::tuple{-camp::get<i>(x)...};
}

/**
 * @tparam T the types stored in the tuple y
 * @param x a tuple of values
 * @brief return a tuple of values defined by applying the unary minus operator to each element of x
 */
template <typename... T>
SERAC_HOST_DEVICE constexpr auto operator-(const camp::tuple<T...>& x)
{
  return unary_minus_helper(x, std::make_integer_sequence<int, static_cast<int>(sizeof...(T))>());
}

/**
 * @brief A helper function for the / operator of tuples
 *
 * @tparam S the types stored in the tuple x
 * @tparam T the types stored in the tuple y
 * @tparam i The integer sequence to i
 * @param x tuple of values
 * @param y tuple of values
 * @return the returned tuple ratio
 */
template <typename... S, typename... T, int... i>
SERAC_HOST_DEVICE constexpr auto div_helper(const camp::tuple<S...>& x, const camp::tuple<T...>& y,
                                            std::integer_sequence<int, i...>)
{
  return camp::tuple{camp::get<i>(x) / camp::get<i>(y)...};
}

/**
 * @tparam S the types stored in the tuple x
 * @tparam T the types stored in the tuple y
 * @param x a tuple of values
 * @param y a tuple of values
 * @brief return a tuple of values defined by elementwise division of x by y
 */
template <typename... S, typename... T>
SERAC_HOST_DEVICE constexpr auto operator/(const camp::tuple<S...>& x, const camp::tuple<T...>& y)
{
  static_assert(sizeof...(S) == sizeof...(T));
  return div_helper(x, y, std::make_integer_sequence<int, static_cast<int>(sizeof...(S))>());
}

/**
 * @brief A helper function for the / operator of tuples
 *
 * @tparam T the types stored in the tuple y
 * @tparam i The integer sequence to i
 * @param x tuple of values
 * @param a the constant numerator
 * @return the returned tuple ratio
 */
template <typename... T, int... i>
SERAC_HOST_DEVICE constexpr auto div_helper(const double a, const camp::tuple<T...>& x, std::integer_sequence<int, i...>)
{
  return camp::tuple{a / camp::get<i>(x)...};
}

/**
 * @brief A helper function for the / operator of tuples
 *
 * @tparam T the types stored in the tuple y
 * @tparam i The integer sequence to i
 * @param x tuple of values
 * @param a the constant denomenator
 * @return the returned tuple ratio
 */
template <typename... T, int... i>
SERAC_HOST_DEVICE constexpr auto div_helper(const camp::tuple<T...>& x, const double a, std::integer_sequence<int, i...>)
{
  return camp::tuple{camp::get<i>(x) / a...};
}

/**
 * @tparam T the types stored in the tuple x
 * @param a the numerator
 * @param x a tuple of denominator values
 * @brief return a tuple of values defined by division of a by the elements of x
 */
template <typename... T>
SERAC_HOST_DEVICE constexpr auto operator/(const double a, const camp::tuple<T...>& x)
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
SERAC_HOST_DEVICE constexpr auto operator/(const camp::tuple<T...>& x, const double a)
{
  return div_helper(x, a, std::make_integer_sequence<int, static_cast<int>(sizeof...(T))>());
}

/**
 * @brief A helper function for the * operator of tuples
 *
 * @tparam S the types stored in the tuple x
 * @tparam T the types stored in the tuple y
 * @tparam i The integer sequence to i
 * @param x tuple of values
 * @param y tuple of values
 * @return the returned tuple product
 */
template <typename... S, typename... T, int... i>
SERAC_HOST_DEVICE constexpr auto mult_helper(const camp::tuple<S...>& x, const camp::tuple<T...>& y,
                                             std::integer_sequence<int, i...>)
{
  return camp::tuple{camp::get<i>(x) * camp::get<i>(y)...};
}

/**
 * @tparam S the types stored in the tuple x
 * @tparam T the types stored in the tuple y
 * @param x a tuple of values
 * @param y a tuple of values
 * @brief return a tuple of values defined by elementwise multiplication of x and y
 */
template <typename... S, typename... T>
SERAC_HOST_DEVICE constexpr auto operator*(const camp::tuple<S...>& x, const camp::tuple<T...>& y)
{
  static_assert(sizeof...(S) == sizeof...(T));
  return mult_helper(x, y, std::make_integer_sequence<int, static_cast<int>(sizeof...(S))>());
}

/**
 * @brief A helper function for the * operator of tuples
 *
 * @tparam T the types stored in the tuple y
 * @tparam i The integer sequence to i
 * @param x tuple of values
 * @param a a constant multiplier
 * @return the returned tuple product
 */
template <typename... T, int... i>
SERAC_HOST_DEVICE constexpr auto mult_helper(const double a, const camp::tuple<T...>& x, std::integer_sequence<int, i...>)
{
  return camp::tuple{a * camp::get<i>(x)...};
}

/**
 * @brief A helper function for the * operator of tuples
 *
 * @tparam T the types stored in the tuple y
 * @tparam i The integer sequence to i
 * @param x tuple of values
 * @param a a constant multiplier
 * @return the returned tuple product
 */
template <typename... T, int... i>
SERAC_HOST_DEVICE constexpr auto mult_helper(const camp::tuple<T...>& x, const double a, std::integer_sequence<int, i...>)
{
  return camp::tuple{camp::get<i>(x) * a...};
}

/**
 * @tparam T the types stored in the tuple
 * @param a a scaling factor
 * @param x the tuple object
 * @brief multiply each component of x by the value a on the left
 */
template <typename... T>
SERAC_HOST_DEVICE constexpr auto operator*(const double a, const camp::tuple<T...>& x)
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
SERAC_HOST_DEVICE constexpr auto operator*(const camp::tuple<T...>& x, const double a)
{
  return mult_helper(x, a, std::make_integer_sequence<int, static_cast<int>(sizeof...(T))>());
}

/**
 * @tparam T the types stored in the tuple
 * @tparam i a list of indices used to acces each element of the tuple
 * @param out the ostream to write the output to
 * @param A the tuple of values
 * @brief helper used to implement printing a tuple of values
 */
template <typename... T, std::size_t... i>
auto& print_helper(std::ostream& out, const camp::tuple<T...>& A, std::integer_sequence<size_t, i...>)
{
  out << "tuple{";
  (..., (out << (i == 0 ? "" : ", ") << camp::get<i>(A)));
  out << "}";
  return out;
}

/**
 * @tparam T the types stored in the tuple
 * @param out the ostream to write the output to
 * @param A the tuple of values
 * @brief print a tuple of values
 */
template <typename... T>
auto& operator<<(std::ostream& out, const camp::tuple<T...>& A)
{
  return print_helper(out, A, std::make_integer_sequence<size_t, sizeof...(T)>());
}

/**
 * @brief A helper to apply a lambda to a tuple
 *
 * @tparam lambda The functor type
 * @tparam T The tuple types
 * @tparam i The integer sequence to i
 * @param f The functor to apply to the tuple
 * @param args The input tuple
 * @return The functor output
 */
template <typename lambda, typename... T, int... i>
SERAC_HOST_DEVICE auto apply_helper(lambda f, camp::tuple<T...>& args, std::integer_sequence<int, i...>)
{
  return f(camp::get<i>(args)...);
}

/**
 * @tparam lambda a callable type
 * @tparam T the types of arguments to be passed in to f
 * @param f the callable object
 * @param args a tuple of arguments
 * @brief a way of passing an n-tuple to a function that expects n separate arguments
 *
 *   e.g. foo(bar, baz) is equivalent to apply(foo, camp::tuple(bar,baz));
 */
template <typename lambda, typename... T>
SERAC_HOST_DEVICE auto apply(lambda f, camp::tuple<T...>& args)
{
  return apply_helper(f, std::move(args), std::make_integer_sequence<int, static_cast<int>(sizeof...(T))>());
}

/**
 * @overload
 */
template <typename lambda, typename... T, int... i>
SERAC_HOST_DEVICE auto apply_helper(lambda f, const camp::tuple<T...>& args, std::integer_sequence<int, i...>)
{
  return f(camp::get<i>(args)...);
}

/**
 * @tparam lambda a callable type
 * @tparam T the types of arguments to be passed in to f
 * @param f the callable object
 * @param args a tuple of arguments
 * @brief a way of passing an n-tuple to a function that expects n separate arguments
 *
 *   e.g. foo(bar, baz) is equivalent to apply(foo, camp::tuple(bar,baz));
 */
template <typename lambda, typename... T>
SERAC_HOST_DEVICE auto apply(lambda f, const camp::tuple<T...>& args)
{
  return apply_helper(f, std::move(args), std::make_integer_sequence<int, static_cast<int>(sizeof...(T))>());
}

/**
 * @brief a struct used to determine the type at index I of a tuple
 *
 * @note see: https://en.cppreference.com/w/cpp/utility/tuple/tuple_element
 *
 * @tparam I the index of the desired type
 * @tparam T a tuple of different types
 */
template <size_t I, class T>
struct tuple_element;

// recursive case
/// @overload
template <size_t I, class Head, class... Tail>
struct tuple_element<I, camp::tuple<Head, Tail...>> : tuple_element<I - 1, camp::tuple<Tail...>> {
};

// base case
/// @overload
template <class Head, class... Tail>
struct tuple_element<0, camp::tuple<Head, Tail...>> {
  using type = Head;  ///< the type at the specified index
};

}  // namespace serac

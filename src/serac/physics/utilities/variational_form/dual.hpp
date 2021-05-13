// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file dual.hpp
 *
 * @brief This file contains the declaration of a dual number class
 */

#pragma once

#include <iostream>

#include <cmath>

namespace serac {

/**
 * @brief Dual number struct (value plus gradient)
 * @tparam gradient_type The type of the gradient
 */
template <typename gradient_type>
struct dual {
  double        value;
  gradient_type gradient;
};

template <typename T>
dual(double, T) -> dual<T>;

template <typename gradient_type>
constexpr auto operator+(dual<gradient_type> a, double b)
{
  return dual{a.value + b, a.gradient};
}

template <typename gradient_type>
constexpr auto operator+(double a, dual<gradient_type> b)
{
  return dual{a + b.value, b.gradient};
}

template <typename gradient_type_a, typename gradient_type_b>
constexpr auto operator+(dual<gradient_type_a> a, dual<gradient_type_b> b)
{
  return dual{a.value + b.value, a.gradient + b.gradient};
}

template <typename gradient_type>
constexpr auto operator-(dual<gradient_type> x)
{
  return dual{-x.value, -x.gradient};
}

template <typename gradient_type>
constexpr auto operator-(dual<gradient_type> a, double b)
{
  return dual{a.value - b, a.gradient};
}

template <typename gradient_type>
constexpr auto operator-(double a, dual<gradient_type> b)
{
  return dual{a - b.value, -b.gradient};
}

template <typename gradient_type_a, typename gradient_type_b>
constexpr auto operator-(dual<gradient_type_a> a, dual<gradient_type_b> b)
{
  return dual{a.value - b.value, a.gradient - b.gradient};
}

template <typename gradient_type>
constexpr auto operator*(const dual<gradient_type>& a, double b)
{
  return dual{a.value * b, a.gradient * b};
}

template <typename gradient_type>
constexpr auto operator*(double a, const dual<gradient_type>& b)
{
  return dual{a * b.value, a * b.gradient};
}

template <typename gradient_type_a, typename gradient_type_b>
constexpr auto operator*(dual<gradient_type_a> a, dual<gradient_type_b> b)
{
  return dual{a.value * b.value, b.value * a.gradient + a.value * b.gradient};
}

template <typename gradient_type>
constexpr auto operator/(const dual<gradient_type>& a, double b)
{
  return dual{a.value / b, a.gradient / b};
}

template <typename gradient_type>
constexpr auto operator/(double a, const dual<gradient_type>& b)
{
  return dual{a / b.value, -(a / (b.value * b.value)) * b.gradient};
}

template <typename gradient_type_a, typename gradient_type_b>
constexpr auto operator/(dual<gradient_type_a> a, dual<gradient_type_b> b)
{
  return dual{a.value / b.value, (a.gradient / b.value) - (a.value * b.gradient) / (b.value * b.value)};
}

/**
 * @brief Generates const + non-const overloads for a binary comparison operator
 * Comparisons are conducted against the "value" part of the dual number
 * @param[in] x The comparison operator to overload
 */
#define binary_comparator_overload(x)                           \
  template <typename T>                                         \
  constexpr bool operator x(const dual<T>& a, double b)         \
  {                                                             \
    return a.value x b;                                         \
  }                                                             \
                                                                \
  template <typename T>                                         \
  constexpr bool operator x(double a, const dual<T>& b)         \
  {                                                             \
    return a x b.value;                                         \
  };                                                            \
                                                                \
  template <typename T, typename U>                             \
  constexpr bool operator x(const dual<T>& a, const dual<U>& b) \
  {                                                             \
    return a.value x b.value;                                   \
  };

binary_comparator_overload(<);
binary_comparator_overload(<=);
binary_comparator_overload(==);
binary_comparator_overload(>=);
binary_comparator_overload(>);

#undef binary_comparator_overload

template <typename gradient_type>
constexpr auto& operator+=(dual<gradient_type>& a, const dual<gradient_type>& b)
{
  a.value += b.value;
  a.gradient += b.gradient;
  return a;
}

template <typename gradient_type>
constexpr auto& operator-=(dual<gradient_type>& a, const dual<gradient_type>& b)
{
  a.value -= b.value;
  a.gradient -= b.gradient;
  return a;
}

template <typename gradient_type>
constexpr auto& operator+=(dual<gradient_type>& a, double b)
{
  a.value += b;
  return a;
}

template <typename gradient_type>
constexpr auto& operator-=(dual<gradient_type>& a, double b)
{
  a.value -= b;
  return a;
}

template <typename gradient_type>
auto abs(dual<gradient_type> x)
{
  return (x.value > 0) ? x : -x;
}

template <typename gradient_type>
auto sqrt(dual<gradient_type> x)
{
  using std::sqrt;
  return dual<gradient_type>{sqrt(x.value), x.gradient / (2.0 * sqrt(x.value))};
}

template <typename gradient_type>
auto cos(dual<gradient_type> a)
{
  using std::cos, std::sin;
  return dual<gradient_type>{cos(a.value), -a.gradient * sin(a.value)};
}

template <typename gradient_type>
auto exp(dual<gradient_type> a)
{
  using std::exp;
  return dual<gradient_type>{exp(a.value), exp(a.value)};
}

template <typename gradient_type>
auto log(dual<gradient_type> a)
{
  using std::log;
  return dual<gradient_type>{log(a.value), a.gradient / a.value};
}

template <typename gradient_type>
auto pow(dual<gradient_type> a, dual<gradient_type> b)
{
  using std::pow, std::log;
  double value = pow(a.value, b.value);
  return dual<gradient_type>{value, value * (a.gradient * (b.value / a.value) + b.gradient * log(a.value))};
}

template <typename gradient_type>
auto pow(double a, dual<gradient_type> b)
{
  using std::pow, std::log;
  double value = pow(a, b.value);
  return dual<gradient_type>{value, value * b.gradient * log(a)};
}

template <typename gradient_type>
auto pow(dual<gradient_type> a, double b)
{
  using std::pow;
  double value = pow(a.value, b);
  return dual<gradient_type>{value, value * a.gradient * b / a.value};
}

template <typename T, int... n>
auto& operator<<(std::ostream& out, dual<T> A)
{
  out << '(' << A.value << ' ' << A.gradient << ')';
  return out;
}

constexpr auto make_dual(double x) { return dual{x, 1.0}; }

template <typename T>
auto get_value(const T& arg)
{
  return arg;
}

template <typename T>
auto get_value(dual<T> arg)
{
  return arg.value;
}

template <typename gradient_type>
auto get_gradient(dual<gradient_type> arg)
{
  return arg.gradient;
}

template <typename T>
struct is_dual_number {
  static constexpr bool value = false;
};

template <typename T>
struct is_dual_number<dual<T> > {
  static constexpr bool value = true;
};

}  // namespace serac

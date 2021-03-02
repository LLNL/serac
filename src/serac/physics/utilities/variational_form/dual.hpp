#pragma once

#include <iostream>

#include <cmath>

template < typename gradient_type >
struct dual {
  double value;
  gradient_type gradient;
};

template < typename T >
dual(double, T) -> dual<T>;

template < typename gradient_type >
constexpr auto operator+(dual <gradient_type> a, double b) {
  return dual{a.value+b, a.gradient};
}

template < typename gradient_type >
constexpr auto operator+(double a, dual <gradient_type> b) {
  return dual{a+b.value, b.gradient};
}

template < typename gradient_type_a, typename gradient_type_b >
constexpr auto operator+(dual <gradient_type_a> a, dual<gradient_type_b> b) {
  return dual{a.value + b.value, a.gradient + b.gradient};
}

template < typename gradient_type >
constexpr auto operator-(dual <gradient_type> a, double b) {
  return dual{a.value-b, a.gradient};
}

template < typename gradient_type >
constexpr auto operator-(double a, dual <gradient_type> b) {
  return dual{a-b.value, -b.gradient};
}

template < typename gradient_type_a, typename gradient_type_b >
constexpr auto operator-(dual <gradient_type_a> a, dual<gradient_type_b> b) {
  return dual{a.value - b.value, a.gradient - b.gradient};
}

template < typename gradient_type >
constexpr auto operator*(const dual <gradient_type> & a, double b) {
  return dual{a.value * b, a.gradient * b};
}

template < typename gradient_type >
constexpr auto operator*(double a, const dual <gradient_type> & b) {
  return dual{a * b.value, a * b.gradient};
}

template < typename gradient_type_a, typename gradient_type_b >
constexpr auto operator*(dual <gradient_type_a> a, dual<gradient_type_b> b) {
  return dual{a.value * b.value, b.value * a.gradient + a.value * b.gradient};
}

template < typename gradient_type >
constexpr auto operator/(const dual <gradient_type> & a, double b) {
  return dual{a.value / b, a.gradient / b};
}

template < typename gradient_type >
constexpr auto operator/(double a, const dual <gradient_type> & b) {
  return dual{a / b.value, - (a / (b.value * b.value)) * b.gradient};
}

template < typename gradient_type_a, typename gradient_type_b >
constexpr auto operator/(dual <gradient_type_a> a, dual<gradient_type_b> b) {
  return dual{a.value / b.value, (a.gradient / b.value) - (a.value * b.gradient) / (b.value * b.value)};
}

template < typename gradient_type >
constexpr auto & operator+=(dual <gradient_type> & a, const dual<gradient_type> & b) {
  a.value += b.value;
  a.gradient += b.gradient;
  return a;
}

template < typename gradient_type >
constexpr auto & operator-=(dual <gradient_type> & a, const dual<gradient_type> & b) {
  a.value -= b.value;
  a.gradient -= b.gradient;
  return a;
}

template < typename gradient_type >
constexpr auto & operator+=(dual <gradient_type> & a, double b) {
  a.value += b;
  return a;
}

template < typename gradient_type >
constexpr auto & operator-=(dual <gradient_type> & a, double b) {
  a.value -= b;
  return a;
}

template < typename gradient_type >
auto sqrt(dual <gradient_type> x) {
  return dual<gradient_type>{sqrt(x.value), x.gradient / (2.0 * sqrt(x.value))};
}

template <typename gradient_type>
auto cos(dual<gradient_type> a)
{
  return dual<gradient_type>{cos(a.value), -a.gradient * sin(a.value)};
}

template <typename gradient_type>
auto exp(dual<gradient_type> a)
{
  return dual<gradient_type>{exp(a.value), exp(a.value)};
}

template <typename gradient_type>
auto log(dual<gradient_type> a)
{
  return dual<gradient_type>{log(a.value), a.gradient / a.value};
}

template <typename gradient_type>
auto pow(dual<gradient_type> a, dual<gradient_type> b)
{
  double value = pow(a.value, b.value);
  return dual<gradient_type>{value, value * (a.gradient * (b.value / a.value) + b.gradient * log(a.value))};
}

template <typename gradient_type>
auto pow(double a, dual<gradient_type> b)
{
  double value = pow(a, b.value);
  return dual<gradient_type>{value, value * b.gradient * log(a)};
}

template <typename gradient_type>
auto pow(dual<gradient_type> a, double b)
{
  double value = pow(a.value, b);
  return dual<gradient_type>{value, value * a.gradient * b / a.value};
}

template <typename T, int... n>
auto& operator<<(std::ostream& out, dual<T> A) {
  out << '(' << A.value << ' ' << A.gradient << ')';
  return out;
}

constexpr auto make_dual(double x) { return dual{x, 1.0}; }

template < typename T >
struct is_dual_number {
  static constexpr bool value = false;
};

template < typename T >
struct is_dual_number< dual < T > >{
  static constexpr bool value = true;
};
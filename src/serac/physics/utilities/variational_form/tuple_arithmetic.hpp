#pragma once

#include <tuple>

#include "tensor.hpp"


/////////////////////////////////////////////////

template < typename ... S, typename ... T, int ... I >
constexpr auto plus_impl(const std::tuple< S ... > & A, const std::tuple< T ... > & B, std::integer_sequence<int, I...>) {
    return std::make_tuple((std::get<I>(A) + std::get<I>(B)) ...);
}

template < typename ... S, typename ... T > 
constexpr auto operator+(const std::tuple< S ... > & A, const std::tuple< T ... > & B) {
  static_assert(sizeof ... (S) == sizeof ... (T), "Error in operator+: tuple sizes must match");
  return plus_impl(A, B, std::make_integer_sequence<int, sizeof ... (S)>{});
}

template < typename ... S, typename ... T, int ... I >
constexpr auto minus_impl(const std::tuple< S ... > & A, const std::tuple< T ... > & B, std::integer_sequence<int, I...>) {
    return std::make_tuple((std::get<I>(A) - std::get<I>(B)) ...);
}

template < typename ... S, typename ... T > 
constexpr auto operator-(const std::tuple< S ... > & A, const std::tuple< T ... > & B) {
  static_assert(sizeof ... (S) == sizeof ... (T), "Error in operator+: tuple sizes must match");
  return minus_impl(A, B, std::make_integer_sequence<int, sizeof ... (S)>{});
}

template < typename ... S, typename T, int ... I >
constexpr auto mult_impl(const std::tuple< S ... > & A, T scale, std::integer_sequence<int, I...>) {
    return std::make_tuple(std::get<I>(A) * scale ...);
}

template < typename ... S, typename T > 
constexpr auto operator*(const std::tuple< S ... > & A, T scale) {
  return mult_impl(A, scale, std::make_integer_sequence<int, sizeof ... (S)>{});
}

template < typename S, typename ... T, int ... I >
constexpr auto mult_impl(S scale, const std::tuple< T ... > & A, std::integer_sequence<int, I...>) {
  return std::make_tuple((scale * std::get<I>(A)) ...);
}

template < typename S, typename ... T > 
constexpr auto operator*(S scale, const std::tuple< T ... > & A) {
  return mult_impl(scale, A, std::make_integer_sequence<int, sizeof ... (T)>{});
}


template < int I, int ... i >
constexpr auto make_dual_helper(double arg, std::integer_sequence<int, i...>){
  using gradient_type = std::tuple<
    typename std::conditional< i == I, double, zero >::type ...
  >;
  dual < gradient_type > arg_dual{};
  arg_dual.value = arg;
  std::get<I>(arg_dual.gradient) = 1.0;
  return arg_dual;
}

template < int I, typename T, int ... n, int ... i >
constexpr auto make_dual_helper(tensor< T, n...> arg, std::integer_sequence<int, i...>){
  using gradient_type = std::tuple<
    typename std::conditional< i == I, tensor< T, n...>, zero >::type ...
  >;
  tensor < dual < gradient_type >, n... > arg_dual{};
  for_constexpr<n...>([&](auto ... j){
    arg_dual(j...).value = arg(j...);
    std::get<I>(arg_dual(j...).gradient)(j...) = 1.0;
  });
  return arg_dual;
}

template < typename ... T, int ... i >
constexpr auto make_dual_helper(std::tuple< T ... > args, std::integer_sequence<int, i...> seq){
  return std::make_tuple(
    (make_dual_helper<i>(std::get<i>(args), seq))...
  );
}

template < typename ... T >
constexpr auto make_dual(T ... args){
  return make_dual_helper(std::tuple{args...}, std::make_integer_sequence<int, sizeof...(T)>{});
}

template < typename ... T >
constexpr auto make_dual(std::tuple < T ... > args){
  return make_dual_helper(args, std::make_integer_sequence<int, sizeof...(T)>{});
}
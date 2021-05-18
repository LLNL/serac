#pragma once

#include <tuple>
#include <utility>
#include <type_traits>

namespace
{
    template <typename, template <typename...> typename>
    struct is_instance_helper : public std::false_type {};

    template <template <typename...> typename T, typename...args>
    struct is_instance_helper<T<args...>, T> : public std::true_type {};
}

// see https://stackoverflow.com/a/61040973
template <typename T, template <typename ...> typename U>
using is_instance = is_instance_helper<std::decay_t<T>, U>;

template <class... T>
constexpr bool always_false = false;

// return the first argument in a variadic list
template < typename ... T >
constexpr auto first(T ... args) {
  return std::get< 0 >(std::tuple{args...});
}

// return the last argument in a variadic list
template < typename ... T >
constexpr auto last(T ... args) {
  return std::get< sizeof ... (T) - 1 >(std::tuple{args...});
}

// return the Ith entry in a std::integer_sequence
template < int I, int ... n >
constexpr auto get(std::integer_sequence<int, n...>) {
  constexpr int values[sizeof...(n)] = {n ...};
  return values[I]; 
}

template < int r, int ... n, int ... i >
constexpr auto remove_helper(std::integer_sequence<int,n...>, std::integer_sequence<int,i...>) {
  return std::integer_sequence<int, get<i+(i>=r)>(std::integer_sequence<int,n ...>{}) ... >{};
}

// return a new std::integer_sequence, after removing the rth entry from another std::integer_sequence
template < int r, int ... n >
constexpr auto remove(std::integer_sequence<int,n...> seq) {
  return remove_helper<r>(seq, std::make_integer_sequence< int, int(sizeof ... (n)) - 1 >{});
}

// return the concatenation of two std::integer_sequences
template < int ... n1, int ... n2 >
constexpr auto join(std::integer_sequence<int,n1...>, std::integer_sequence<int,n2...>) {
  return std::integer_sequence<int,n1...,n2...>{};
}

namespace detail {
  template < typename lambda, int ... i >
  inline constexpr void for_constexpr(lambda && f, std::integral_constant< int, i > ... args) {
    f(args ...);
  }

  template < int ... n, typename lambda, typename ... arg_types >
  inline constexpr void for_constexpr(lambda && f, std::integer_sequence< int, n ... >, arg_types ... args) {
    (detail::for_constexpr(f, args ..., std::integral_constant< int, n >{}), ...);
  }
}

// multidimensional loop tool that evaluates the lambda body
// inside the innermost loop. The number of integer template
// describe the number of nested loops, and the actual values
// describe the dimensions of each loop. e.g.
//
// for_constexpr< 2, 3 >([](auto i, auto j) { std::cout << i << " " << j << std::endl; }
//
// will print
// 0 0
// 0 1
// 0 2
// 1 0
// 1 1
// 1 2
// 
// latter integer template parameters correspond to more nested loops
// 
// The lambda function should be a callable object taking sizeof ... (n) arguments.
// Anything returned from f() will be discarded
//
// note: this forces multidimensional loop unrolling, which can be beneficial for
// runtime performance, but can hurt compile time and executable size as the loop
// dimensions become larger.
template < int ... n, typename lambda >
inline constexpr void for_constexpr(lambda && f) {
  detail::for_constexpr(f, std::make_integer_sequence<int, n>{} ...);
}

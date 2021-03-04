#pragma once

#include <utility>

namespace impl {
  template < typename lambda, int ... i >
  inline constexpr void for_constexpr(lambda && f, std::integral_constant< int, i > ... args) {
    f(args ...);
  }

  template < int ... n, typename lambda, typename ... arg_types >
  inline constexpr void for_constexpr(lambda && f, std::integer_sequence< int, n ... >, arg_types ... args) {
    (impl::for_constexpr(f, args ..., std::integral_constant< int, n >{}), ...);
  }
}

template < int ... n, typename lambda >
inline constexpr void for_constexpr(lambda && f) {
  impl::for_constexpr(f, std::make_integer_sequence<int, n>{} ...);
}
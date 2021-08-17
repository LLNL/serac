#pragma once

#include <utility>

#include "serac/infrastructure/accelerator.hpp"

namespace serac {

// This is a class that mimics most of std::tuple's interface, except that it is usable in CUDA kernels and 
// admits some arithmetic operator overloads
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

template <typename S, typename... T>
SERAC_HOST_DEVICE constexpr auto& get(tuple<T...>& values)
{
  if constexpr (sizeof...(T) > 0) {
    if constexpr (std::is_same_v<S, decltype(values.v0)>) {
      return values.v0;
    }
  }
  if constexpr (sizeof...(T) > 1) {
    if constexpr (std::is_same_v<S, decltype(values.v1)>) {
      return values.v1;
    }
  }
  if constexpr (sizeof...(T) > 2) {
    if constexpr (std::is_same_v<S, decltype(values.v2)>) {
      return values.v2;
    }
  }
  if constexpr (sizeof...(T) > 3) {
    if constexpr (std::is_same_v<S, decltype(values.v3)>) {
      return values.v3;
    }
  }
  if constexpr (sizeof...(T) > 4) {
    if constexpr (std::is_same_v<S, decltype(values.v4)>) {
      return values.v4;
    }
  }
  if constexpr (sizeof...(T) > 5) {
    if constexpr (std::is_same_v<S, decltype(values.v5)>) {
      return values.v5;
    }
  }
  if constexpr (sizeof...(T) > 6) {
    if constexpr (std::is_same_v<S, decltype(values.v6)>) {
      return values.v6;
    }
  }
  if constexpr (sizeof...(T) > 7) {
    if constexpr (std::is_same_v<S, decltype(values.v7)>) {
      return values.v7;
    }
  }
}

template <typename S, typename... T>
SERAC_HOST_DEVICE constexpr auto get(const tuple<T...>& values)
{
  if constexpr (sizeof...(T) > 0) {
    if constexpr (std::is_same_v<S, decltype(values.v0)>) {
      return values.v0;
    }
  }
  if constexpr (sizeof...(T) > 1) {
    if constexpr (std::is_same_v<S, decltype(values.v1)>) {
      return values.v1;
    }
  }
  if constexpr (sizeof...(T) > 2) {
    if constexpr (std::is_same_v<S, decltype(values.v2)>) {
      return values.v2;
    }
  }
  if constexpr (sizeof...(T) > 3) {
    if constexpr (std::is_same_v<S, decltype(values.v3)>) {
      return values.v3;
    }
  }
  if constexpr (sizeof...(T) > 4) {
    if constexpr (std::is_same_v<S, decltype(values.v4)>) {
      return values.v4;
    }
  }
  if constexpr (sizeof...(T) > 5) {
    if constexpr (std::is_same_v<S, decltype(values.v5)>) {
      return values.v5;
    }
  }
  if constexpr (sizeof...(T) > 6) {
    if constexpr (std::is_same_v<S, decltype(values.v6)>) {
      return values.v6;
    }
  }
  if constexpr (sizeof...(T) > 7) {
    if constexpr (std::is_same_v<S, decltype(values.v7)>) {
      return values.v7;
    }
  }
}

template <typename... S, typename... T, int... i>
SERAC_HOST_DEVICE constexpr auto plus_helper(const tuple<S...>& x, const tuple<T...>& y,
                                             std::integer_sequence<int, i...>)
{
  return tuple{get<i>(x) + get<i>(y)...};
}

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

template <typename... T>
SERAC_HOST_DEVICE constexpr auto operator/(const double a, const tuple<T...>& x)
{
  return div_helper(a, x, std::make_integer_sequence<int, static_cast<int>(sizeof...(T))>());
}

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

template <typename... T>
SERAC_HOST_DEVICE constexpr auto operator*(const double a, const tuple<T...>& x)
{
  return mult_helper(a, x, std::make_integer_sequence<int, static_cast<int>(sizeof...(T))>());
}

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

template <typename lambda, typename... T>
SERAC_HOST_DEVICE auto apply(lambda f, const tuple<T...>& args)
{
  return apply_helper(f, std::move(args), std::make_integer_sequence<int, static_cast<int>(sizeof...(T))>());
}

#if 0
template <class T>
struct unwrap_refwrapper
{
    using type = T;
};
 
template <class T>
struct unwrap_refwrapper<std::reference_wrapper<T>>
{
    using type = T&;
};
 
template <class T>
using unwrap_decay_t = typename unwrap_refwrapper<typename std::decay<T>::type>::type;
// or use std::unwrap_ref_decay_t (since C++20)
 
template <class... Types>
constexpr // since C++14
tuple<unwrap_decay_t<Types>...> make_tuple(Types&&... args)
{
    return tuple<unwrap_decay_t<Types>...>(std::forward<Types>(args)...);
}
#endif

}  // namespace serac

// namespace std {
//
//  template < size_t i, typename ... T >
//  constexpr auto & get(const serac::tuple< T ... > & values) { return serac::get<i>(values); }
//
//  template < size_t i, typename ... T >
//  constexpr auto get(const serac::tuple< T ... > values) { return serac::get<i>(values); }
//
//  template< typename  ... Types >
//  struct tuple_size< serac::tuple<Types...> > : std::integral_constant<std::size_t, sizeof...(Types)> {};
//
//} // namespace std

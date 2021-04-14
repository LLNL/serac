#pragma once

#include "tensor.hpp"
#include "dimensions.hpp"

template <int... n>
struct zero_tensor {
  template <typename T>
  operator tensor<T, n...>()
  {
    return tensor<T, n...>{};
  }
};

template <int... n>
constexpr auto dimensions_of(zero_tensor<n...>)
{
  return Dimensions<n...>{};
}

constexpr auto make_zero_tensor(double) { return zero_tensor<>{}; }

template <int... n>
constexpr auto make_zero_tensor(Dimensions<n...>)
{
  return zero_tensor<n...>{};
}

template <typename T, int... n>
constexpr auto make_zero_tensor(tensor<T, n...>)
{
  return zero_tensor<n...>{};
}

template <int... n>
constexpr auto operator+(zero_tensor<n...>, zero_tensor<n...>)
{
  return zero_tensor<n...>{};
}

template <typename T, int... n>
constexpr auto operator+(zero_tensor<n...>, tensor<T, n...> other)
{
  return other;
}

template <typename T, int... n>
constexpr auto operator+(T other, zero_tensor<n...>)
{
  return other;
}

/////////////////////////////////////////////////

template <int... n>
constexpr auto operator-(zero_tensor<n...>, zero_tensor<n...>)
{
  return zero_tensor<n...>{};
}

template <typename T, int... n>
constexpr auto operator-(zero_tensor<n...>, tensor<T, n...> other)
{
  return -other;
}

template <typename T, int... n>
constexpr auto operator-(tensor<T, n...> other, zero_tensor<n...>)
{
  return other;
}

/////////////////////////////////////////////////

template <int... m, int... n>
constexpr auto dot(zero_tensor<m...> A, zero_tensor<n...> B)
{
  constexpr bool scalar_multiplication = (sizeof...(m) == 0) || (sizeof...(n) == 0);
  if constexpr (scalar_multiplication) {
    return zero_tensor<m..., n...>{};
  } else {
    static_assert((last(Dimensions<m...>{}) == first(Dimensions<n...>{})) || scalar_multiplication);
    return make_zero_tensor(remove_last(dimensions_of(A)) + remove_first(dimensions_of(B)));
  }
}

template <int... m, typename T, int... n>
constexpr auto dot(zero_tensor<m...> A, tensor<T, n...> B)
{
  constexpr bool scalar_multiplication = (sizeof...(m) == 0) || (sizeof...(n) == 0);
  if constexpr (scalar_multiplication) {
    return zero_tensor<m..., n...>{};
  } else {
    static_assert((last(Dimensions<m...>{}) == first(Dimensions<n...>{})) || scalar_multiplication);
    return make_zero_tensor(remove_last(dimensions_of(A)) + remove_first(dimensions_of(B)));
  }
}

template <typename T, int... m, int... n>
constexpr auto dot(tensor<T, m...> A, zero_tensor<n...> B)
{
  constexpr bool scalar_multiplication = (sizeof...(m) == 0) || (sizeof...(n) == 0);
  if constexpr (scalar_multiplication) {
    return zero_tensor<m..., n...>{};
  } else {
    static_assert((last(Dimensions<m...>{}) == first(Dimensions<n...>{})) || scalar_multiplication);
    return make_zero_tensor(remove_last(dimensions_of(A)) + remove_first(dimensions_of(B)));
  }
}

template <typename T, int... n, typename = std::enable_if_t<std::is_arithmetic_v<T> || is_dual_number<T>::value> >
constexpr auto operator*(zero_tensor<n...>, T /*other*/)
{
  return zero_tensor<n...>{};
}

template <typename T, int... n, typename = std::enable_if_t<std::is_arithmetic_v<T> || is_dual_number<T>::value> >
constexpr auto operator*(T /*other*/, zero_tensor<n...>)
{
  return zero_tensor<n...>{};
}
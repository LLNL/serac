// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file isotropic_tensor.hpp
 *
 * @brief Implementation of isotropic tensor classes
 */

#pragma once

#include <iostream>
#include <type_traits>

namespace serac {

template <int n>
struct always_false : std::false_type {
};

template <typename T, int... n>
struct isotropic_tensor;

template <typename T, int n>
struct isotropic_tensor<T, n> {
  static_assert(always_false<n>{}, "error: there is no such thing as a rank-1 isotropic tensor!");
};

// rank-2 isotropic tensors are just identity matrices
template <typename T, int m>
struct isotropic_tensor<T, m, m> {
  SERAC_HOST_DEVICE constexpr T operator()(int i, int j) const { return (i == j) * value; }
  T value;
};

template <int m>
SERAC_HOST_DEVICE constexpr isotropic_tensor<double, m, m> Identity()
{
  return isotropic_tensor<double, m, m>{1.0};
}

template <typename S, typename T, int m>
SERAC_HOST_DEVICE constexpr auto operator*(S scale, isotropic_tensor<T, m, m> I)
{
  return isotropic_tensor<decltype(S{} * T{}), m, m>{I.value * scale};
}

template <typename S, typename T, int m>
SERAC_HOST_DEVICE constexpr auto operator*(isotropic_tensor<T, m, m> I, S scale)
{
  return isotropic_tensor<decltype(S{}, T{}), m, m>{I.value * scale};
}

template <typename S, typename T, int m>
SERAC_HOST_DEVICE constexpr auto operator+(isotropic_tensor<S, m, m> I1, isotropic_tensor<T, m, m> I2)
{
  return isotropic_tensor<decltype(S{} + T{}), m, m>{I1.value + I2.value};
}

template <typename S, typename T, int m>
SERAC_HOST_DEVICE constexpr auto operator-(isotropic_tensor<S, m, m> I1, isotropic_tensor<T, m, m> I2)
{
  return isotropic_tensor<decltype(S{} - T{}), m, m>{I1.value - I2.value};
}

template <typename S, typename T, int m>
SERAC_HOST_DEVICE constexpr auto operator+(const isotropic_tensor<S, m, m>& I, const tensor<T, m, m>& A)
{
  tensor<decltype(S{} + T{}), m, m> output{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      output[i][j] = I.value * (i == j) + A[i][j];
    }
  }
  return output;
}

template <typename S, typename T, int m>
SERAC_HOST_DEVICE constexpr auto operator+(const tensor<S, m, m>& A, const isotropic_tensor<T, m, m>& I)
{
  tensor<decltype(S{} + T{}), m, m> output{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      output[i][j] = A[i][j] + I.value * (i == j);
    }
  }
  return output;
}

template <typename S, typename T, int m>
SERAC_HOST_DEVICE constexpr auto operator-(const isotropic_tensor<S, m, m>& I, const tensor<T, m, m>& A)
{
  tensor<decltype(S{} - T{}), m, m> output{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      output[i][j] = I.value * (i == j) - A[i][j];
    }
  }
  return output;
}

template <typename S, typename T, int m>
SERAC_HOST_DEVICE constexpr auto operator-(const tensor<S, m, m>& A, const isotropic_tensor<T, m, m>& I)
{
  tensor<decltype(S{} - T{}), m, m> output{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      output[i][j] = A[i][j] - I.value * (i == j);
    }
  }
  return output;
}

template <typename S, typename T, int m, int... n>
SERAC_HOST_DEVICE constexpr auto dot(const isotropic_tensor<S, m, m>& I, const tensor<T, m, n...>& A)
{
  return I.value * A;
}

template <typename S, typename T, int m, int... n>
SERAC_HOST_DEVICE constexpr auto dot(const tensor<S, n...>& A, isotropic_tensor<T, m, m> I)
{
  constexpr int dimensions[sizeof...(n)] = {n...};
  static_assert(dimensions[sizeof...(n) - 1] == m);
  return A * I.value;
}

template <typename S, typename T, int m, int... n>
SERAC_HOST_DEVICE constexpr auto ddot(const isotropic_tensor<S, m, m>& I, const tensor<T, m, m>& A)
{
  return I.value * tr(A);
}

template <typename T, int m>
SERAC_HOST_DEVICE constexpr auto sym(const isotropic_tensor<T, m, m>& I)
{
  return I;
}

template <typename T, int m>
SERAC_HOST_DEVICE constexpr auto antisym(const isotropic_tensor<T, m, m>&)
{
  return zero{};
}

template <typename T, int m>
SERAC_HOST_DEVICE constexpr auto tr(const isotropic_tensor<T, m, m>& I)
{
  return I.value * m;
}

template <typename T, int m>
SERAC_HOST_DEVICE constexpr auto transpose(const isotropic_tensor<T, m, m>& I)
{
  return I;
}

template <typename T, int m>
SERAC_HOST_DEVICE constexpr auto det(const isotropic_tensor<T, m, m>& I)
{
  return std::pow(I.value, m);
}

template <typename T, int m>
SERAC_HOST_DEVICE constexpr auto norm(const isotropic_tensor<T, m, m>& I)
{
  return sqrt(I.value * I.value * m);
}

template <typename T, int m>
SERAC_HOST_DEVICE constexpr auto sqnorm(const isotropic_tensor<T, m, m>& I)
{
  return I.value * I.value * m;
}

// rank-3 isotropic tensors are just the alternating symbol
template <typename T>
struct isotropic_tensor<T, 3, 3, 3> {
  SERAC_HOST_DEVICE constexpr T operator()(int i, int j, int k) const { return 0.5 * (i - j) * (j - k) * (k - i) * value; }
  T value;
};

// there are 3 linearly-independent rank-4 isotropic tensors,
// so the general one will be some linear combination of them
template <typename T, int m>
struct isotropic_tensor<T, m, m, m, m> {
  T c1, c2, c3;

  SERAC_HOST_DEVICE constexpr T operator()(int i, int j, int k, int l) const
  {
    return c1 * (i == j) * (k == l) + c2 * ((i == k) * (j == l) + (i == l) * (j == k)) * 0.5 +
           c3 * ((i == k) * (j == l) - (i == l) * (j == k)) * 0.5;
  }
};

template <int m>
SERAC_HOST_DEVICE constexpr auto SymmetricIdentity()
{
  return isotropic_tensor<double, m, m, m, m>{0.0, 1.0, 0.0};
}

template <int m>
SERAC_HOST_DEVICE constexpr auto AntisymmetricIdentity()
{
  return isotropic_tensor<double, m, m, m, m>{0.0, 0.0, 1.0};
}

template <typename S, typename T, int m>
SERAC_HOST_DEVICE constexpr auto operator*(S scale, isotropic_tensor<T, m, m, m, m> I)
{
  return isotropic_tensor<decltype(S{} * T{}), m, m, m, m>{I.c1 * scale, I.c2 * scale, I.c3 * scale};
}

template <typename S, typename T, int m>
SERAC_HOST_DEVICE constexpr auto operator*(isotropic_tensor<S, m, m, m, m> I, T scale)
{
  return isotropic_tensor<decltype(S{} * T{}), m, m, m, m>{I.c1 * scale, I.c2 * scale, I.c3 * scale};
}

template <typename S, typename T, int m>
SERAC_HOST_DEVICE constexpr auto operator+(isotropic_tensor<S, m, m, m, m> I1, isotropic_tensor<T, m, m, m, m> I2)
{
  return isotropic_tensor<decltype(S{} + T{}), m, m, m, m>{I1.c1 + I2.c1, I1.c2 + I2.c2, I1.c3 + I2.c3};
}

template <typename S, typename T, int m>
SERAC_HOST_DEVICE constexpr auto operator-(isotropic_tensor<S, m, m, m, m> I1, isotropic_tensor<T, m, m, m, m> I2)
{
  return isotropic_tensor<decltype(S{} - T{}), m, m, m, m>{I1.c1 - I2.c1, I1.c2 - I2.c2, I1.c3 - I2.c3};
}

template <typename S, typename T, int m, int... n>
SERAC_HOST_DEVICE constexpr auto ddot(const isotropic_tensor<S, m, m, m, m>& I, const tensor<T, m, m, n...>& A)
{
  return I.c1 * tr(A) * Identity<m>() + I.c2 * sym(A) + I.c3 * antisym(A);
}

}  // namespace serac

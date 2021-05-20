// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "host_device_annotations.hpp"

#include "dual.hpp"
#include "array.hpp"

#include "detail/metaprogramming.hpp"

namespace serac {

namespace detail {
template <typename T, typename i0_t>
SERAC_HOST_DEVICE constexpr auto get(const T& values, i0_t i0)
{
  return values[i0];
}

template <typename T, typename i0_t, typename i1_t>
SERAC_HOST_DEVICE constexpr auto get(const T& values, i0_t i0, i1_t i1)
{
  return values[i0][i1];
}

template <typename T, typename i0_t, typename i1_t, typename i2_t>
SERAC_HOST_DEVICE constexpr auto get(const T& values, i0_t i0, i1_t i1, i2_t i2)
{
  return values[i0][i1][i2];
}

template <typename T, typename i0_t, typename i1_t, typename i2_t, typename i3_t>
SERAC_HOST_DEVICE constexpr auto get(const T& values, i0_t i0, i1_t i1, i2_t i2, i3_t i3)
{
  return values[i0][i1][i2][i3];
}

template <typename T, typename i0_t>
SERAC_HOST_DEVICE constexpr auto& get(T& values, i0_t i0)
{
  return values[i0];
}

template <typename T, typename i0_t, typename i1_t>
SERAC_HOST_DEVICE constexpr auto& get(T& values, i0_t i0, i1_t i1)
{
  return values[i0][i1];
}

template <typename T, typename i0_t, typename i1_t, typename i2_t>
SERAC_HOST_DEVICE constexpr auto& get(T& values, i0_t i0, i1_t i1, i2_t i2)
{
  return values[i0][i1][i2];
}

template <typename T, typename i0_t, typename i1_t, typename i2_t, typename i3_t>
SERAC_HOST_DEVICE constexpr auto& get(T& values, i0_t i0, i1_t i1, i2_t i2, i3_t i3)
{
  return values[i0][i1][i2][i3];
}
}  // namespace detail

/// @cond
template <typename T, int... n>
struct tensor;

template <typename T>
struct tensor<T> {
  using type                    = T;
  static constexpr int ndim     = 0;
  static constexpr int shape[1] = {0};

  SERAC_HOST_DEVICE constexpr auto& operator()(array<int, ndim>) { return value; }
  SERAC_HOST_DEVICE constexpr auto  operator()(array<int, ndim>) const { return value; }

  template <typename... S>
  SERAC_HOST_DEVICE constexpr auto& operator()(S...)
  {
    return value;
  }

  template <typename... S>
  SERAC_HOST_DEVICE constexpr auto operator()(S...) const
  {
    return value;
  }

  SERAC_HOST_DEVICE tensor() : value{} {}
  SERAC_HOST_DEVICE tensor(T v) : value(v) {}
  SERAC_HOST_DEVICE operator T() { return value; }
  T                 value;
};

template <typename T, int n>
struct tensor<T, n> {
  using type                       = T;
  static constexpr int ndim        = 1;
  static constexpr int shape[ndim] = {n};

  template <typename S>
  SERAC_HOST_DEVICE constexpr auto& operator()(S i)
  {
    return detail::get(value, i);
  }

  template <typename S>
  SERAC_HOST_DEVICE constexpr auto operator()(S i) const
  {
    return detail::get(value, i);
  }

  SERAC_HOST_DEVICE constexpr auto& operator[](int i) { return value[i]; };
  SERAC_HOST_DEVICE constexpr auto  operator[](int i) const { return value[i]; };
  T                                 value[n];
};
/// @endcond

/**
 * @brief Arbitrary-rank tensor class
 * @tparam T The scalar type of the tensor
 * @tparam first The leading dimension of the tensor
 * @tparam last The parameter pack of the remaining dimensions
 */
template <typename T, int first, int... rest>
struct tensor<T, first, rest...> {
  /**
   * @brief The scalar type
   */
  using type = T;
  /**
   * @brief The rank of the tensor
   */
  static constexpr int ndim = 1 + sizeof...(rest);
  /**
   * @brief The array of dimensions containing the shape (not the data itself)
   * Similar to numpy.ndarray.shape
   */
  static constexpr int shape[ndim] = {first, rest...};

  /**
   * @brief Retrieves the sub-tensor corresponding to the indices provided in the pack @a i
   * @param[in] i The pack of indices
   */
  template <typename... S>
  SERAC_HOST_DEVICE constexpr auto& operator()(S... i)
  {
    // FIXME: Compile-time check for <= 4 indices??
    return detail::get(value, i...);
  };
  /// @overload
  template <typename... S>
  SERAC_HOST_DEVICE constexpr auto operator()(S... i) const
  {
    return detail::get(value, i...);
  };

  /**
   * @brief Retrieves the "row" of the tensor corresponding to index @a i
   * @param[in] i The index to retrieve a rank - 1 tensor from
   */
  SERAC_HOST_DEVICE constexpr auto& operator[](int i) { return value[i]; };
  /// @overload
  SERAC_HOST_DEVICE constexpr auto operator[](int i) const { return value[i]; };

  /**
   * @brief The actual tensor data
   */
  tensor<T, rest...> value[first];
};

template <typename T, int n1>
tensor(const T (&data)[n1]) -> tensor<T, n1>;

template <typename T, int n1, int n2>
tensor(const T (&data)[n1][n2]) -> tensor<T, n1, n2>;

/**
 * @brief A sentinel struct for eliding no-op tensor operations
 */
struct zero {
  SERAC_HOST_DEVICE operator double() { return 0.0; }

  template <typename T, int... n>
  SERAC_HOST_DEVICE operator tensor<T, n...>()
  {
    return tensor<T, n...>{};
  }

  template <typename... T>
  SERAC_HOST_DEVICE auto operator()(T...)
  {
    return zero{};
  }

  template <typename T>
  SERAC_HOST_DEVICE auto operator=(T)
  {
    return zero{};
  }
};

SERAC_HOST_DEVICE constexpr auto operator+(zero, zero) { return zero{}; }

template <typename T>
SERAC_HOST_DEVICE constexpr auto operator+(zero, T other)
{
  return other;
}

template <typename T>
SERAC_HOST_DEVICE constexpr auto operator+(T other, zero)
{
  return other;
}

/////////////////////////////////////////////////

SERAC_HOST_DEVICE constexpr auto operator-(zero, zero) { return zero{}; }

template <typename T>
SERAC_HOST_DEVICE constexpr auto operator-(zero, T other)
{
  return -other;
}

template <typename T>
SERAC_HOST_DEVICE constexpr auto operator-(T other, zero)
{
  return other;
}

/////////////////////////////////////////////////

SERAC_HOST_DEVICE constexpr auto operator*(zero, zero) { return zero{}; }

template <typename T>
SERAC_HOST_DEVICE constexpr auto operator*(zero, T /*other*/)
{
  return zero{};
}

template <typename T>
SERAC_HOST_DEVICE constexpr auto operator*(T /*other*/, zero)
{
  return zero{};
}

/**
 * @brief Removes 1s from tensor dimensions
 * For example, a tensor<T, 1, 10> is equivalent to a tensor<T, 10>
 * @tparam T The scalar type of the tensor
 * @tparam n1 The first dimension
 * @tparam n2 The second dimension
 */
template <typename T, int n1, int n2 = 1>
using reduced_tensor = std::conditional_t<
    (n1 == 1 && n2 == 1), double,
    std::conditional_t<n1 == 1, tensor<T, n2>, std::conditional_t<n2 == 1, tensor<T, n1>, tensor<T, n1, n2>>>>;

/**
 * @brief Returns a @p IndexSpace object that can be used
 * to iterate over a tensor of arbitary dimension
 */
template <typename T, int... n>
SERAC_HOST_DEVICE constexpr auto indices(tensor<T, n...>)
{
  return IndexSpace<n...>{};
}

/**
 * @brief Creates a tensor given the dimensions in a @p std::integer_sequence
 * @see std::integer_sequence
 * @tparam n The parameter pack of integer dimensions
 */
template <typename T, int... n>
SERAC_HOST_DEVICE constexpr auto tensor_with_shape(std::integer_sequence<int, n...>)
{
  return tensor<T, n...>{};
}

namespace detail {
template <int n>
using always_int = int;
}  // namespace detail

/**
 * @brief Creates a tensor of requested dimension by subsequent calls to a functor
 * Can be thought of as analogous to @p std::transform in that the set of possible
 * indices for dimensions @p n are transformed into the values of the tensor by @a f
 * @tparam n The parameter pack of integer dimensions
 * @param[in] f The functor to generate the tensor values from
 * @pre @a f must accept @p sizeof...(n) arguments of type @p int
 */
template <int... n, typename lambda_type>
SERAC_HOST_DEVICE constexpr auto make_tensor(lambda_type f)
{
  using T = decltype(f(n...));
  tensor<T, n...> A{};
  if constexpr (sizeof...(n) == 0) {
    A.value = f();
  } else {
    for_constexpr<n...>([&](auto... i) { A(i...) = f(i...); });
  }
  return A;
}

template <typename S, typename T, int... n>
SERAC_HOST_DEVICE constexpr auto operator+(const tensor<S, n...>& A, const tensor<T, n...>& B)
{
  tensor<decltype(S{} + T{}), n...> C{};
  for (int i = 0; i < tensor<T, n...>::shape[0]; i++) {
    C[i] = A[i] + B[i];
  }
  return C;
}

template <typename T, int... n>
SERAC_HOST_DEVICE constexpr auto operator-(const tensor<T, n...>& A)
{
  tensor<T, n...> B{};
  for (int i = 0; i < tensor<T, n...>::shape[0]; i++) {
    B[i] = -A[i];
  }
  return B;
}

template <typename S, typename T, int... n>
SERAC_HOST_DEVICE constexpr auto operator-(const tensor<S, n...>& A, const tensor<T, n...>& B)
{
  tensor<decltype(S{} + T{}), n...> C{};
  for (int i = 0; i < tensor<T, n...>::shape[0]; i++) {
    C[i] = A[i] - B[i];
  }
  return C;
}

template <typename S, typename T, int... n,
          typename = std::enable_if_t<std::is_arithmetic_v<S> || is_dual_number<S>::value>>
SERAC_HOST_DEVICE constexpr auto operator*(S scale, const tensor<T, n...>& A)
{
  tensor<decltype(S{} * T{}), n...> C{};
  for (int i = 0; i < tensor<T, n...>::shape[0]; i++) {
    C[i] = scale * A[i];
  }
  return C;
}

template <typename S, typename T, int... n,
          typename = std::enable_if_t<std::is_arithmetic_v<S> || is_dual_number<S>::value>>
SERAC_HOST_DEVICE constexpr auto operator*(const tensor<T, n...>& A, S scale)
{
  tensor<decltype(T{} * S{}), n...> C{};
  for (int i = 0; i < tensor<T, n...>::shape[0]; i++) {
    C[i] = A[i] * scale;
  }
  return C;
}

template <typename S, typename T, int... n,
          typename = std::enable_if_t<std::is_arithmetic_v<S> || is_dual_number<S>::value>>
constexpr auto operator/(S scale, const tensor<T, n...>& A)
{
  tensor<decltype(S{} * T{}), n...> C{};
  for (int i = 0; i < tensor<T, n...>::shape[0]; i++) {
    C[i] = scale / A[i];
  }
  return C;
}

template <typename S, typename T, int... n,
          typename = std::enable_if_t<std::is_arithmetic_v<S> || is_dual_number<S>::value>>
constexpr auto operator/(const tensor<T, n...>& A, S scale)
{
  tensor<decltype(T{} * S{}), n...> C{};
  for (int i = 0; i < tensor<T, n...>::shape[0]; i++) {
    C[i] = A[i] / scale;
  }
  return C;
}

template <typename S, typename T, int... n>
constexpr auto& operator+=(tensor<S, n...>& A, const tensor<T, n...>& B)
{
  for (int i = 0; i < tensor<S, n...>::shape[0]; i++) {
    A[i] += B[i];
  }
  return A;
}

template <typename T, int... n>
constexpr auto& operator+=(tensor<T, n...>& A, zero)
{
  return A;
}

template <typename S, typename T, int... n>
constexpr auto& operator-=(tensor<S, n...>& A, const tensor<T, n...>& B)
{
  for (int i = 0; i < tensor<S, n...>::shape[0]; i++) {
    A[i] -= B[i];
  }
  return A;
}

template <typename T, int... n>
constexpr auto& operator-=(tensor<T, n...>& A, zero)
{
  return A;
}

template <typename S, typename T>
constexpr auto outer(S A, T B)
{
  static_assert(std::is_arithmetic_v<S> && std::is_arithmetic_v<T>,
                "outer product types must be tensor or arithmetic_type");
  return A * B;
}

template <typename S, typename T, int n>
constexpr auto outer(S A, tensor<T, n> B)
{
  static_assert(std::is_arithmetic_v<S>, "outer product types must be tensor or arithmetic_type");
  tensor<decltype(S{} * T{}), n> AB{};
  for (int i = 0; i < n; i++) {
    AB[i] = A * B[i];
  }
  return AB;
}

template <typename S, typename T, int m>
constexpr auto outer(const tensor<S, m>& A, T B)
{
  static_assert(std::is_arithmetic_v<T>, "outer product types must be tensor or arithmetic_type");
  tensor<decltype(S{} * T{}), m> AB{};
  for (int i = 0; i < m; i++) {
    AB[i] = A[i] * B;
  }
  return AB;
}

template <typename T, int n>
constexpr auto outer(zero, const tensor<T, n>&)
{
  return zero{};
}

template <typename T, int n>
constexpr auto outer(const tensor<T, n>&, zero)
{
  return zero{};
}

template <typename S, typename T, int m, int n>
constexpr auto outer(S A, const tensor<T, m, n>& B)
{
  static_assert(std::is_arithmetic_v<S>, "outer product types must be tensor or arithmetic_type");
  tensor<decltype(S{} * T{}), m, n> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      AB[i][j] = A * B[i][j];
    }
  }
  return AB;
}

template <typename S, typename T, int m, int n>
constexpr auto outer(const tensor<S, m>& A, const tensor<T, n>& B)
{
  tensor<decltype(S{} * T{}), m, n> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      AB[i][j] = A[i] * B[j];
    }
  }
  return AB;
}

template <typename S, typename T, int m, int n>
constexpr auto outer(const tensor<S, m, n>& A, T B)
{
  static_assert(std::is_arithmetic_v<T>, "outer product types must be tensor or arithmetic_type");
  tensor<decltype(S{} * T{}), m, n> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      AB[i][j] = A[i][j] * B;
    }
  }
  return AB;
}

template <typename S, typename T, int m, int n, int p>
constexpr auto outer(const tensor<S, m, n>& A, const tensor<T, p>& B)
{
  tensor<decltype(S{} * T{}), m, n, p> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < p; k++) {
        AB[i][j][k] = A[i][j] * B[k];
      }
    }
  }
  return AB;
}

template <typename S, typename T, int m, int n, int p>
constexpr auto outer(const tensor<S, m>& A, const tensor<T, n, p>& B)
{
  tensor<decltype(S{} * T{}), m, n, p> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < p; k++) {
        AB[i][j][k] = A[i] * B[j][k];
      }
    }
  }
  return AB;
}

template <typename S, typename T, int m, int n, int p, int q>
constexpr auto outer(const tensor<S, m, n>& A, const tensor<T, p, q>& B)
{
  tensor<decltype(S{} * T{}), m, n, p, q> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < p; k++) {
        for (int l = 0; l < q; l++) {
          AB[i][j][k][l] = A[i][j] * B[k][l];
        }
      }
    }
  }
  return AB;
}

template <typename S, typename T, int m, int n>
constexpr auto inner(const tensor<S, m, n>& A, const tensor<T, m, n>& B)
{
  decltype(S{} * T{}) sum{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      sum += A[i][j] * B[i][j];
    }
  }
  return sum;
}

template <typename S, typename T, int m, int n, int p>
constexpr auto dot(const tensor<S, m, n>& A, const tensor<T, n, p>& B)
{
  tensor<decltype(S{} * T{}), m, p> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < p; j++) {
      for (int k = 0; k < n; k++) {
        AB[i][j] = AB[i][j] + A[i][k] * B[k][j];
      }
    }
  }
  return AB;
}

template <typename S, typename T, int m, int n>
constexpr auto dot(const tensor<S, m>& A, const tensor<T, m, n>& B)
{
  tensor<decltype(S{} * T{}), n> AB{};
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      AB[i] = AB[i] + A[j] * B[j][i];
    }
  }
  return AB;
}

template <typename S, typename T, int m, int n>
constexpr auto dot(const tensor<S, m, n>& A, const tensor<T, n>& B)
{
  tensor<decltype(S{} * T{}), m> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      AB[i] = AB[i] + A[i][j] * B[j];
    }
  }
  return AB;
}

template <typename S, typename T, int n>
constexpr auto dot(const tensor<S, n>& A, const tensor<T, n>& B)
{
  decltype(S{} * T{}) AB{};
  for (int i = 0; i < n; i++) {
    AB += A[i] * B[i];
  }
  return AB;
}

template <typename S, typename T, int m, int n, int p>
constexpr auto dot(const tensor<S, m, n, p>& A, const tensor<T, p>& B)
{
  tensor<decltype(S{} * T{}), m, n> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < p; k++) {
        AB[i][j] += A[i][j][k] * B[k];
      }
    }
  }
  return AB;
}

template <typename S, typename T, typename U, int m, int n>
constexpr auto dot(const tensor<S, m>& u, const tensor<T, m, n>& A, const tensor<U, n>& v)
{
  decltype(S{} * T{} * U{}) uAv{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      uAv += u[i] * A[i][j] * v[j];
    }
  }
  return uAv;
}

template <typename S, typename T, int m, int n, int p, int q>
constexpr auto ddot(const tensor<S, m, n, p, q>& A, const tensor<T, p, q>& B)
{
  tensor<decltype(S{} * T{}), m, n> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < p; k++) {
        for (int l = 0; l < q; l++) {
          AB[i][j] += A[i][j][k][l] * B[k][l];
        }
      }
    }
  }
  return AB;
}

template <typename S, typename T, int m, int n, int p>
constexpr auto ddot(const tensor<S, m, n, p>& A, const tensor<T, n, p>& B)
{
  tensor<decltype(S{} * T{}), m> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < p; k++) {
        AB[i] += A[i][j][k] * B[j][k];
      }
    }
  }
  return AB;
}

template <typename S, typename T, int m, int n>
constexpr auto ddot(const tensor<S, m, n>& A, const tensor<T, m, n>& B)
{
  decltype(S{} * T{}) AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      AB += A[i][j] * B[i][j];
    }
  }
  return AB;
}

template <typename S, typename T, int m, int n, int p>
constexpr auto operator*(const tensor<S, m, n>& A, const tensor<T, n, p>& B)
{
  return dot(A, B);
}

template <typename S, typename T, int m, int n>
constexpr auto operator*(const tensor<S, m>& A, const tensor<T, m, n>& B)
{
  return dot(A, B);
}

template <typename S, typename T, int m, int n>
constexpr auto operator*(const tensor<S, m, n>& A, const tensor<T, n>& B)
{
  return dot(A, B);
}

/**
 * @brief Returns the squared Frobenius norm of the tensor
 * @param[in] A The tensor to obtain the squared norm from
 */
template <typename T, int m>
constexpr auto sqnorm(const tensor<T, m>& A)
{
  T total{};
  for (int i = 0; i < m; i++) {
    total += A[i] * A[i];
  }
  return total;
}
/// @overload
template <typename T, int m, int n>
constexpr auto sqnorm(const tensor<T, m, n>& A)
{
  T total{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      total += A[i][j] * A[i][j];
    }
  }
  return total;
}
/// @overload
template <typename T, int... n>
constexpr auto sqnorm(const tensor<T, n...>& A)
{
  T total{};
  for_constexpr<n...>([&](auto... i) { total += A(i...) * A(i...); });
  return total;
}

/**
 * @brief Returns the Frobenius norm of the tensor
 * @param[in] A The tensor to obtain the norm from
 */
template <typename T, int... n>
auto norm(const tensor<T, n...>& A)
{
  using std::sqrt;
  return sqrt(sqnorm(A));
}

/**
 * @brief Normalizes the tensor
 * Each element is divided by the Frobenius norm of the tensor, @see norm
 * @param[in] A The tensor to normalize
 */
template <typename T, int... n>
auto normalize(const tensor<T, n...>& A)
{
  return A / norm(A);
}

/**
 * @brief Returns the trace of a square matrix
 * @param[in] A The matrix to compute the trace of
 * @return The sum of the elements on the main diagonal
 */
template <typename T, int n>
constexpr auto tr(const tensor<T, n, n>& A)
{
  T trA{};
  for (int i = 0; i < n; i++) {
    trA = trA + A[i][i];
  }
  return trA;
}

/**
 * @brief Returns the symmetric part of a square matrix
 * @param[in] A The matrix to obtain the symmetric part of
 * @return (1/2) * (A + A^T)
 */
template <typename T, int n>
constexpr auto sym(const tensor<T, n, n>& A)
{
  tensor<T, n, n> symA{};
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      symA[i][j] = 0.5 * (A[i][j] + A[j][i]);
    }
  }
  return symA;
}

/**
 * @brief Calculates the deviator of a matrix (rank-2 tensor)
 * @param[in] A The matrix to calculate the deviator of
 * In the context of stress tensors, the deviator is obtained by
 * subtracting the mean stress (average of main diagonal elements)
 * from each element on the main diagonal
 */
template <typename T, int n>
constexpr auto dev(const tensor<T, n, n>& A)
{
  auto devA = A;
  auto trA  = tr(A);
  for (int i = 0; i < n; i++) {
    devA[i][i] -= trA / n;
  }
  return devA;
}

/**
 * @brief Obtains the identity matrix of the specified dimension
 * @return I_dim
 */
template <int dim>
constexpr tensor<double, dim, dim> Identity()
{
  tensor<double, dim, dim> I{};
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      I[i][j] = (i == j);
    }
  }
  return I;
}

/**
 * @brief Returns the transpose of the matrix
 * @param[in] A The matrix to obtain the transpose of
 */
template <typename T, int m, int n>
constexpr auto transpose(const tensor<T, m, n>& A)
{
  tensor<T, n, m> AT{};
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      AT[i][j] = A[j][i];
    }
  }
  return AT;
}

/**
 * @brief Returns the determinant of a matrix
 * @param[in] A The matrix to obtain the determinant of
 */
template <typename T>
constexpr auto det(const tensor<T, 2, 2>& A)
{
  return A[0][0] * A[1][1] - A[0][1] * A[1][0];
}
/// @overload
template <typename T>
constexpr auto det(const tensor<T, 3, 3>& A)
{
  return A[0][0] * A[1][1] * A[2][2] + A[0][1] * A[1][2] * A[2][0] + A[0][2] * A[1][0] * A[2][1] -
         A[0][0] * A[1][2] * A[2][1] - A[0][1] * A[1][0] * A[2][2] - A[0][2] * A[1][1] * A[2][0];
}

/**
 * @brief Solves Ax = b for x using Gaussian elimination with partial pivoting
 * @param[in] A The coefficient matrix A
 * @param[in] b The righthand side vector b
 * @note @a A and @a b are by-value as they are mutated as part of the elimination
 */
template <typename T, int n>
constexpr tensor<T, n> linear_solve(tensor<T, n, n> A, const tensor<T, n> b)
{
  constexpr auto abs  = [](double x) { return (x < 0) ? -x : x; };
  constexpr auto swap = [](auto& x, auto& y) {
    auto tmp = x;
    x        = y;
    y        = tmp;
  };

  tensor<double, n> x{};

  for (int i = 0; i < n; i++) {
    // Search for maximum in this column
    double max_val = abs(A[i][i]);

    int max_row = i;
    for (int j = i + 1; j < n; j++) {
      if (abs(A[j][i]) > max_val) {
        max_val = abs(A[j][i]);
        max_row = j;
      }
    }

    swap(b[max_row], b[i]);
    swap(A[max_row], A[i]);

    // zero entries below in this column
    for (int j = i + 1; j < n; j++) {
      double c = -A[j][i] / A[i][i];
      A[j] += c * A[i];
      b[j] += c * b[i];
      A[j][i] = 0;
    }
  }

  // Solve equation Ax=b for an upper triangular matrix A
  for (int i = n - 1; i >= 0; i--) {
    x[i] = b[i] / A[i][i];
    for (int j = i - 1; j >= 0; j--) {
      b[j] -= A[j][i] * x[i];
    }
  }

  return x;
}

/**
 * @brief Inverts a matrix
 * @param[in] A The matrix to invert
 * @note Uses a shortcut for inverting a 2-by-2 matrix
 */
constexpr tensor<double, 2, 2> inv(const tensor<double, 2, 2>& A)
{
  double inv_detA(1.0 / det(A));

  tensor<double, 2, 2> invA{};

  invA[0][0] = A[1][1] * inv_detA;
  invA[0][1] = -A[0][1] * inv_detA;
  invA[1][0] = -A[1][0] * inv_detA;
  invA[1][1] = A[0][0] * inv_detA;

  return invA;
}

/**
 * @overload
 * @note Uses a shortcut for inverting a 3-by-3 matrix
 */
constexpr tensor<double, 3, 3> inv(const tensor<double, 3, 3>& A)
{
  double inv_detA(1.0 / det(A));

  tensor<double, 3, 3> invA{};

  invA[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * inv_detA;
  invA[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) * inv_detA;
  invA[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * inv_detA;
  invA[1][0] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) * inv_detA;
  invA[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * inv_detA;
  invA[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) * inv_detA;
  invA[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) * inv_detA;
  invA[2][1] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) * inv_detA;
  invA[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * inv_detA;

  return invA;
}
/**
 * @overload
 * @note For N-by-N matrices with N > 3, requires Gaussian elimination
 * with partial pivoting
 */
template <typename T, int n>
constexpr tensor<T, n, n> inv(const tensor<T, n, n>& A)
{
  constexpr auto abs  = [](double x) { return (x < 0) ? -x : x; };
  constexpr auto swap = [](auto& x, auto& y) {
    auto tmp = x;
    x        = y;
    y        = tmp;
  };

  tensor<double, n, n> B = Identity<n>();

  for (int i = 0; i < n; i++) {
    // Search for maximum in this column
    double max_val = abs(A[i][i]);

    int max_row = i;
    for (int j = i + 1; j < n; j++) {
      if (abs(A[j][i]) > max_val) {
        max_val = abs(A[j][i]);
        max_row = j;
      }
    }

    swap(B[max_row], B[i]);
    swap(A[max_row], A[i]);

    // zero entries below in this column
    for (int j = i + 1; j < n; j++) {
      if (A[j][i] != 0.0) {
        // if (A[j][i] * A[j][i] > 1.0e-25) {
        double c = -A[j][i] / A[i][i];
        A[j] += c * A[i];
        B[j] += c * B[i];
        A[j][i] = 0;
      }
    }
  }

  // upper triangular solve
  for (int i = n - 1; i >= 0; i--) {
    B[i] = B[i] / A[i][i];
    for (int j = i - 1; j >= 0; j--) {
      if (A[j][i] != 0.0) {
        // if (A[j][i] * A[j][i] > 1.0e-25) {
        B[j] -= A[j][i] * B[i];
      }
    }
  }

  return B;
}

// hardcode the analytic derivative of the
// inverse of a square matrix, rather than
// apply gauss elimination directly on the dual number types
template <typename gradient_type, int n>
auto inv(tensor<dual<gradient_type>, n, n> A)
{
  auto invA = inv(get_value(A));
  return make_tensor<n, n>([&](int i, int j) {
    auto          value = invA[i][j];
    gradient_type gradient{};
    for (int k = 0; k < n; k++) {
      for (int l = 0; l < n; l++) {
        gradient -= invA[i][k] * A[k][l].gradient * invA[l][j];
      }
    }
    return dual<gradient_type>{value, gradient};
  });
}

template <typename T1, int... n1, typename T2, int... n2>
constexpr auto outer_product(const tensor<T1, n1...>& A, const tensor<T2, n2...>& B)
{
  tensor<decltype(T1{} * T2{}), n1..., n2...> AB{};
  for_constexpr<n1...>([&](auto... i1) {
    for_constexpr<n2...>([&](auto... i2) { AB({i1..., i2...}) = A({i1...}) * B({i2...}); });
  });
  return AB;
}

template <int J, typename T1, int... I1, typename T2, int... I2, int... I1H, int... I2T>
constexpr auto dot_product_helper(const tensor<T1, I1...>& A, const tensor<T2, I2...>& B,
                                  std::integer_sequence<int, I1H...>, std::integer_sequence<int, I2T...>)
{
  tensor<decltype(T1{} * T2{}), I1H..., I2T...> AB{};
  for_constexpr<I1H...>([&](auto... i1) {
    for_constexpr<I2T...>([&, i1...](auto... i2) {
      for (int j = 0; j < J; j++) {
        AB({i1..., i2...}) += A({i1..., j}) * B({j, i2...});
      }
    });
  });
  return AB;
}

template <typename T1, int... I1, typename T2, int... I2>
constexpr auto dot_product(const tensor<T1, I1...>& A, const tensor<T2, I2...>& B)
{
  static_assert(last(I1...) == first(I2...), "error: dimension mismatch");
  constexpr auto I1H = remove<sizeof...(I1) - 1>(std::integer_sequence<int, I1...>{});
  constexpr auto I2T = remove<0>(std::integer_sequence<int, I2...>{});
  return dot_product_helper<last(I1...)>(A, B, I1H, I2T);
}

template <typename T, int... n>
auto& operator<<(std::ostream& out, const tensor<T, n...>& A)
{
  out << '{' << A[0];
  for (int i = 1; i < tensor<T, n...>::shape[0]; i++) {
    out << ", " << A[i];
  }
  out << '}';
  return out;
}

/**
 * @brief Zeroes out trivially small tensor values
 * @param[in] A The tensor to "chop"
 */
template <int n>
constexpr auto chop(const tensor<double, n>& A)
{
  auto copy = A;
  for (int i = 0; i < n; i++) {
    if (copy[i] * copy[i] < 1.0e-20) {
      copy[i] = 0.0;
    }
  }
  return copy;
}
/// @overload
template <int m, int n>
constexpr auto chop(const tensor<double, m, n>& A)
{
  auto copy = A;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (copy[i][j] * copy[i][j] < 1.0e-20) {
        copy[i][j] = 0.0;
      }
    }
  }
  return copy;
}

/**
 * @brief Constructs a tensor of dual numbers from a tensor of values
 * @param[in] A The tensor of values
 * The gradients for each value will be set to 1
 */
template <int... n>
constexpr auto make_dual(const tensor<double, n...>& A)
{
  tensor<dual<tensor<double, n...>>, n...> A_dual{};
  for_constexpr<n...>([&](auto... i) {
    A_dual(i...).value          = A(i...);
    A_dual(i...).gradient(i...) = 1.0;
  });
  return A_dual;
}

template <typename T>
struct underlying {
  using type = void;
};

template <typename T, int... n>
struct underlying<tensor<T, n...>> {
  using type = T;
};

template <>
struct underlying<double> {
  using type = double;
};

namespace detail {

template <typename T1, typename T2>
struct outer_prod;

template <int... m, int... n>
struct outer_prod<tensor<double, m...>, tensor<double, n...>> {
  using type = tensor<double, m..., n...>;
};

template <int... n>
struct outer_prod<double, tensor<double, n...>> {
  using type = tensor<double, n...>;
};

template <int... n>
struct outer_prod<tensor<double, n...>, double> {
  using type = tensor<double, n...>;
};

template <>
struct outer_prod<double, double> {
  using type = tensor<double>;
};

template <typename T>
struct outer_prod<zero, T> {
  using type = zero;
};

template <typename T>
struct outer_prod<T, zero> {
  using type = zero;
};

}  // namespace detail

template <typename T1, typename T2>
using outer_product_t = typename detail::outer_prod<T1, T2>::type;

/**
 * @brief Retrieves a value tensor from a tensor of dual numbers
 * @param[in] arg The tensor of dual numbers
 */
template <typename T, int... n>
auto get_value(const tensor<dual<T>, n...>& arg)
{
  tensor<double, n...> value{};
  for_constexpr<n...>([&](auto... i) { value(i...) = arg(i...).value; });
  return value;
}

/**
 * @brief Retrieves the gradient component of a double (which is nothing)
 * @return The sentinel, @see zero
 */
auto get_gradient(double /* arg */) { return zero{}; }

/**
 * @brief Retrieves a gradient tensor from a tensor of dual numbers
 * @param[in] arg The tensor of dual numbers
 */
template <typename T, int... n>
auto get_gradient(const tensor<dual<double>, n...>& arg)
{
  tensor<double, n...> g{};
  for_constexpr<n...>([&](auto... i) { g[{i...}] = arg[{i...}].gradient; });
  return g;
}

/// @overload
template <int... n, int... m>
auto get_gradient(const tensor<dual<tensor<double, m...>>, n...>& arg)
{
  tensor<double, n..., m...> g{};
  for_constexpr<n...>([&](auto... i) { g(i...) = arg(i...).gradient; });
  return g;
}

constexpr auto chain_rule(const zero /* df_dx */, const zero /* dx */) { return zero{}; }

template <typename T>
constexpr auto chain_rule(const zero /* df_dx */, const T /* dx */)
{
  return zero{};
}

template <typename T>
constexpr auto chain_rule(const T /* df_dx */, const zero /* dx */)
{
  return zero{};
}

constexpr auto chain_rule(const double df_dx, const double dx) { return df_dx * dx; }

template <int... n>
constexpr auto chain_rule(const tensor<double, n...>& df_dx, const double dx)
{
  return df_dx * dx;
}

template <int... n>
constexpr auto chain_rule(const tensor<double, n...>& df_dx, const tensor<double, n...>& dx)
{
  double total{};
  for_constexpr<n...>([&](auto... i) { total += df_dx(i...) * dx(i...); });
  return total;
}

template <int m, int... n>
constexpr auto chain_rule(const tensor<double, m, n...>& df_dx, const tensor<double, n...>& dx)
{
  tensor<double, m> total{};
  for (int i = 0; i < m; i++) {
    total[i] = chain_rule(df_dx[i], dx);
  }
  return total;
}

template <int m, int n, int... p>
auto chain_rule(const tensor<double, m, n, p...>& df_dx, const tensor<double, p...>& dx)
{
  tensor<double, m, n> total{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      total[i][j] = chain_rule(df_dx[i][j], dx);
    }
  }
  return total;
}

/*
tuple<double, double, double> output = chain_rule(
  tuple<
    tuple<double,double>,
    tuple<double,double>
  >,
  tuple<double, double>
);

tuple<double, double, double> output = chain_rule(
  tuple<
    tuple<double,double>,
    tuple<double,double>,
    tuple<double,double>
  >,
  tuple<double, double>
);

double output = chain_rule(
  tuple< double, double >,
  tuple< double, double >
);
*/

}  // namespace serac

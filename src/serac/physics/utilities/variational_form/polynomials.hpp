#pragma once

#include "tensor.hpp"

namespace serac {

// clang-format off
/**
 * @brief The positions (in 1D space) of Gauss-Lobatto points
 * @tparam n The number of points
 * @param[in] a The left endpoint of the interval
 * @param[in] b The right endpoint of the interval
 */
template <int n, typename T = double >
constexpr tensor<T, n> GaussLobattoNodes(T a = T(0), T b = T(1)) {
  if constexpr (n == 2) return {a, b}; 
  if constexpr (n == 3) return {a, a + 0.5000000000000000 * (b-a), b}; 
  if constexpr (n == 4) return {a, a + 0.2763932022500210 * (b-a), a + 0.7236067977499790 * (b-a), b};
  return tensor<double, n>{};
};

/**
 * @brief The positions (in 1D space) of Gauss-Legendre points
 * @tparam n The number of points
 * @param[in] a The left endpoint of the interval
 * @param[in] b The right endpoint of the interval
 */
template <int n, typename T = double >
constexpr tensor<T, n> GaussLegendreNodes(T a = T(0), T b = T(1)) {
  if constexpr (n == 1) return {a + 0.5000000000000000 * (b-a)};
  if constexpr (n == 2) return {a + 0.2113248654051871 * (b-a), a + 0.7886751345948129 * (b-a)};
  if constexpr (n == 3) return {a + 0.1127016653792583 * (b-a), a + 0.5 * (b-a), a + 0.8872983346207417 * (b-a)}; 
  if constexpr (n == 4) return {a + 0.06943184420297371 * (b-a), a + 0.3300094782075719 * (b-a), a + 0.6699905217924281 * (b-a), a + 0.9305681557970263 * (b-a)};
  return tensor<double, n>{};
};

/**
 * @brief The weights associated with each Gauss-Legendre point
 * @tparam n The number of points
 */
template <int n, typename T = double >
constexpr tensor<T, n> GaussLegendreWeights() {
  if constexpr (n == 1) return {1.000000000000000};
  if constexpr (n == 2) return {0.500000000000000, 0.500000000000000}; 
  if constexpr (n == 3) return {0.277777777777778, 0.444444444444444, 0.277777777777778};
  if constexpr (n == 4) return {0.173927422568727, 0.326072577431273, 0.326072577431273, 0.173927422568727};
  return tensor<double, n>{};
};
// clang-format on

constexpr int factorial(int n)
{
  int nfactorial = 1;
  for (int i = 2; i <= n; i++) {
    nfactorial *= i;
  }
  return nfactorial;
}

template <int n, typename T>
constexpr tensor<T, n> powers(T x)
{
  tensor<T, n> values{};
  values[0] = T(1.0);
  for (int i = 1; i < n; i++) {
    values[i] = values[i - 1] * x;
  }
  return values;
}

/**
 * @brief Chebyshev polynomials of the first kind
 * Satisfying:
 * T_n(cos(t)) == cos(n*t)
 */
template <int n, typename S>
constexpr tensor<S, n> ChebyshevT(S x)
{
  tensor<S, n> T{};

  if (0 < n) T[0] = 1.0;
  if (1 < n) T[1] = x;
  for (int i = 2; i < n; i++) {
    T[i] = 2 * x * T[i - 1] - T[i - 2];
  }

  return T;
}

/**
 * @brief Chebyshev polynomials of the second kind
 * Satisfying:
 * sin(t) U_n(cos(t)) == sin((n+1)*t)
 */
template <int n, typename T>
tensor<T, n> ChebyshevU(T x)
{
  tensor<T, n> U{};

  if (0 < n) U[0] = 1.0;
  if (1 < n) U[1] = 2 * x;
  for (int i = 2; i < n; i++) {
    U[i] = 2 * x * U[i - 1] - U[i - 2];
  }

  return U;
}

/**
 * Legendre Polynomials, orthogonal on the domain (-1, 1)
 * with unit weight function.
 */
template <int n, typename T>
tensor<T, n> Legendre(T x)
{
  tensor<T, n> P{};

  if (0 < n) P[0] = 1.0;
  if (1 < n) P[1] = x;
  for (int i = 2; i < n; i++) {
    P[i] = ((2 * i - 1) * x * P[i - 1] - (i - 1) * P[i - 2]) / T(i);
  }

  return P;
}

template <int n, typename T>
tensor<T, n> Bernstein(T s)
{
  tensor<T, n> B;

  T t = 1.0 - s;

  T f = 1.0;
  for (int i = 0; i < n; i++) {
    B[i] = f;
    f *= s / (i + 1);
  }

  f = factorial(n - 1);
  for (int i = 0; i < n; i++) {
    B[n - i - 1] *= f;
    f *= t / (i + 1);
  }

  return B;
}

template <int n, typename T>
tensor<T, n> GaussLobattoInterpolation(T x)
{
  if constexpr (n == 2) {
    return {0.5 * (1.0 - x), 0.5 * (1.0 + x)};
  }
  if constexpr (n == 3) {
    return {0.5 * x * (x - 1.0), 1.0 - x * x, 0.5 * x * (x + 1.0)};
  }
  if constexpr (n == 4) {
    static constexpr double s = 2.23606797749978981;
    return {-0.125 * (x - 1.0) * (5.0 * x * x - 1.0), 0.625 * (s * x - 1.0) * (x * x - 1.0),
            -0.625 * (s * x + 1.0) * (x * x - 1.0), 0.125 * (x + 1.0) * (5.0 * x * x - 1.0)};
  }

  return tensor<T, n>{};
}

template <int n, typename T>
tensor<T, n> GaussLobattoInterpolationDerivative(T x)
{
  if constexpr (n == 2) {
    return {-0.5, 0.5};
  }
  if constexpr (n == 3) {
    return {x - 0.5, -2.0 * x, x + 0.5};
  }
  if constexpr (n == 4) {
    static constexpr double s = 2.23606797749978981;
    return {0.125 * (1.0 + 5.0 * (2.0 - 3.0 * x) * x), 0.125 * (-5.0 * (s + x * (2.0 - 3.0 * s * x))),
            0.125 * (5.0 * (s - x * (2.0 + 3.0 * s * x))), 0.125 * (-1.0 + 5.0 * x * (2 + 3.0 * x))};
  }

  return tensor<T, n>{};
}

template <int n, typename T>
constexpr tensor<T, n> GaussLobattoInterpolation01(T x)
{
  if constexpr (n == 2) {
    return {1.0 - x, x};
  }
  if constexpr (n == 3) {
    return {(-1.0 + x) * (-1.0 + 2.0 * x), -4.0 * (-1.0 + x) * x, x * (-1.0 + 2.0 * x)};
  }
  if constexpr (n == 4) {
    constexpr double sqrt5 = 2.23606797749978981;
    return {-(-1.0 + x) * (1.0 + 5.0 * (-1.0 + x) * x), -0.5 * sqrt5 * (5.0 + sqrt5 - 10.0 * x) * (-1.0 + x) * x,
            -0.5 * sqrt5 * (-1.0 + x) * x * (-5.0 + sqrt5 + 10.0 * x), x * (1.0 + 5.0 * (-1.0 + x) * x)};
  }

  return tensor<T, n>{};
}

template <int n, typename T>
constexpr tensor<T, n> GaussLobattoInterpolationDerivative01([[maybe_unused]] T x)
{
  if constexpr (n == 2) {
    return {-1, 1};
  }
  if constexpr (n == 3) {
    return {-3.0 + 4.0 * x, 4.0 - 8.0 * x, -1.0 + 4.0 * x};
  }
  if constexpr (n == 4) {
    constexpr double sqrt5 = 2.23606797749978981;
    return {-6.0 + 5.0 * (4.0 - 3.0 * x) * x, 2.5 * (1.0 + sqrt5 + 2.0 * x * (-1.0 - 3.0 * sqrt5 + 3.0 * sqrt5 * x)),
            -2.5 * (-1.0 + sqrt5 + 2.0 * x * (1.0 - 3.0 * sqrt5 + 3.0 * sqrt5 * x)), 1.0 + 5.0 * x * (-2.0 + 3.0 * x)};
  }

  return tensor<T, n>{};
}

template <int n, typename T>
tensor<T, n> GaussLegendreInterpolation(T x)
{
  if constexpr (n == 1) {
    return {1.0};
  }

  if constexpr (n == 2) {
    return {0.5 - 0.8660254037844387 * x, 0.5 + 0.8660254037844387 * x};
  }

  if constexpr (n == 3) {
    return {-0.645497224367903 * x + 0.833333333333333 * x * x, 1.000000000000000 - 1.666666666666667 * x * x,
            0.6454972243679028 * x + 0.8333333333333333 * x * x};
  }

  return tensor<T, n>{};
}

template <int n, typename T>
tensor<T, n> GaussLegendreInterpolationDerivative(T x)
{
  if constexpr (n == 1) {
    return {0.0};
  }

  if constexpr (n == 2) {
    return {-0.866025403784439, 0.866025403784439};
  }

  if constexpr (n == 3) {
    return {0.1666666666666667 * (-3.872983346207417 + 10.0 * x), -3.333333333333333 * x,
            0.1666666666666667 * (3.872983346207417 + 10.0 * x)};
  }

  return tensor<T, n>{};
}

template <int n, typename T>
tensor<T, n> GaussLegendreInterpolation01([[maybe_unused]] T x)
{
  if constexpr (n == 1) {
    return {1.0};
  }

  if constexpr (n == 2) {
    return {1.36602540378444 - 1.73205080756888 * x, -0.366025403784439 + 1.73205080756888 * x};
  }

  if constexpr (n == 3) {
    return {1.47883055770124 + x * (-4.6243277820691 + 3.3333333333333 * x),
            -0.66666666666667 + x * (6.6666666666667 - 6.6666666666667 * x),
            0.187836108965431 + x * (-2.04233888459753 + 3.3333333333333 * x)};
  }

  return tensor<T, n>{};
}

template <int n, typename T>
tensor<T, n> GaussLegendreInterpolationDerivative01(T x)
{
  if constexpr (n == 1) {
    return {0.0};
  }

  if constexpr (n == 2) {
    return {-1.732050807568877, 1.732050807568877};
  }

  if constexpr (n == 3) {
    return {-4.6243277820691 + 6.6666666666667 * x, 6.6666666666667 - 13.3333333333333 * x,
            -2.04233888459753 + 6.6666666666667 * x};
  }

  return tensor<T, n>{};
}

}  // namespace serac

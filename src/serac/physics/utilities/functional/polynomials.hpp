// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file polynomials.hpp
 *
 * @brief Definitions of 1D quadrature weights and node locations and polynomial basis functions
 */

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
 * 
 * Mathematica/Wolfram Language code to generate more entries in the table:
 * Do[Print["if constexpr (n == " <> ToString[n] <> ") return " <> ToString[GaussianQuadratureWeights[n, 0, 1, 17][[All, 1]]] <> ";"], {n, 1, 8}] 
 */
// clang-format off
template <int n, typename T = double >
constexpr tensor<T, n> GaussLegendreNodes() {
  if constexpr (n == 1) return {0.50000000000000000};
  if constexpr (n == 2) return {0.2113248654051871, 0.7886751345948129};
  if constexpr (n == 3) return {0.1127016653792583, 0.500000000000000, 0.887298334620742};
  if constexpr (n == 4) return {0.0694318442029737, 0.330009478207572, 0.669990521792428, 0.930568155797026};
  if constexpr (n == 5) return {0.0469100770306680, 0.230765344947158, 0.500000000000000, 0.769234655052842, 0.953089922969332};
  if constexpr (n == 6) return {0.0337652428984240, 0.169395306766868, 0.380690406958402, 0.619309593041598, 0.830604693233132, 0.966234757101576};
  if constexpr (n == 7) return {0.0254460438286207, 0.129234407200303, 0.297077424311301, 0.500000000000000, 0.702922575688699, 0.87076559279970, 0.97455395617138};
  if constexpr (n == 8) return {0.0198550717512319, 0.101666761293187, 0.237233795041836, 0.408282678752175, 0.591717321247825, 0.76276620495816, 0.89833323870681, 0.98014492824877};
  return tensor<double, n>{};
};
// clang-format on

/**
 * @brief The weights associated with each Gauss-Legendre point
 * @tparam n The number of points
 *
 * Mathematica/Wolfram Language code to generate more entries in the table:
 * Do[Print["if constexpr (n == " <> ToString[n] <> ") return " <> ToString[GaussianQuadratureWeights[n, 0, 1, 17][[All,
 * 2]]] <> ";"], {n, 1, 8}]
 */
template <int n, typename T = double>
constexpr tensor<T, n> GaussLegendreWeights()
{
  if constexpr (n == 1) return {1.000000000000000};
  if constexpr (n == 2) return {0.500000000000000, 0.500000000000000};
  if constexpr (n == 3) return {0.277777777777778, 0.444444444444444, 0.277777777777778};
  if constexpr (n == 4) return {0.173927422568727, 0.326072577431273, 0.326072577431273, 0.173927422568727};
  if constexpr (n == 5)
    return {0.118463442528095, 0.239314335249683, 0.284444444444444, 0.239314335249683, 0.118463442528095};
  if constexpr (n == 6)
    return {0.085662246189585, 0.180380786524069, 0.233956967286346,
            0.233956967286346, 0.180380786524069, 0.085662246189585};
  if constexpr (n == 7)
    return {0.0647424830844348, 0.139852695744638, 0.190915025252559, 0.208979591836735,
            0.190915025252559,  0.139852695744638, 0.0647424830844348};
  if constexpr (n == 8)
    return {0.0506142681451881, 0.111190517226687, 0.156853322938944, 0.181341891689181,
            0.181341891689181,  0.156853322938944, 0.111190517226687, 0.0506142681451881};
  return tensor<double, n>{};
};
// clang-format on

/**
 * @brief compute n!
 * @param[in] n
 */
constexpr int factorial(int n)
{
  int nfactorial = 1;
  for (int i = 2; i <= n; i++) {
    nfactorial *= i;
  }
  return nfactorial;
}

/**
 * @brief compute the first n powers of x
 * @tparam n how many powers to compute
 * @param[in] x the number to be raised to varying powers
 */
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
 * @tparam n how many entries to compute
 * @param[in] x where to evaluate the polynomials
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
 * @tparam n how many entries to compute
 * @param[in] x where to evaluate the polynomials
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
 * @brief Legendre Polynomials, orthogonal on the domain (-1, 1)
 * with unit weight function.
 * @tparam n how many entries to compute
 * @param[in] x where to evaluate the polynomials
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

/**
 * @brief Bernstein Polynomials on the domain [0, 1]
 * @tparam n how many entries to compute
 * @param[in] s where to evaluate the polynomials
 */
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

/**
 * @brief Lagrange Interpolating polynomials for nodes at Gauss-Lobatto points on the interval [0, 1]
 * @tparam n how many entries to compute
 * @param[in] x where to evaluate the polynomials
 */
template <int n, typename T>
constexpr tensor<T, n> GaussLobattoInterpolation(T x)
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

/**
 * @brief Derivatives of the Lagrange Interpolating polynomials for nodes at Gauss-Lobatto points on the interval [0, 1]
 * @tparam n how many entries to compute
 * @param[in] x where to evaluate the polynomials
 */
template <int n, typename T>
constexpr tensor<T, n> GaussLobattoInterpolationDerivative([[maybe_unused]] T x)
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

/**
 * @brief Lagrange Interpolating polynomials for nodes at Gauss-Legendre points on the interval [0, 1]
 *
 * Do[xi = GaussianQuadratureWeights[n, 0, 1, 30][[All, 1]];
 *  Print["if constexpr (n == " <> ToString[n] <> ") return " <> ToString[CForm /@ HornerForm /@
 * (InterpolatingPolynomial[Transpose[{xi, #}], x] & /@ IdentityMatrix[n])] <> ";"], {n, 1, 4}
 * ]
 *
 * note: the quadratic (n == 2) and cubic (n == 3) versions of this function only satisfied the kronecker delta property
 * to an error of ~1.0e-13 and keeping more than 16 significant figures seems to help bring that error down to machine
 * epsilon again.
 *
 * @tparam n how many entries to compute
 * @param[in] x where to evaluate the polynomials
 */
template <int n, typename T>
tensor<T, n> GaussLegendreInterpolation([[maybe_unused]] T x)
{
  if constexpr (n == 1) return {1};
  if constexpr (n == 2)
    return {1.3660254037844386467637231708 - 1.732050807568877293527446342 * x,
            -0.3660254037844386467637231708 + 1.732050807568877293527446342 * x};
  if constexpr (n == 3)
    return {1.47883055770123614752987757 + x * (-4.6243277820691389617264218 + 3.33333333333333333333333333 * x),
            -0.666666666666666666666666667 + (6.66666666666666666666666667 - 6.666666666666666666666666667 * x) * x,
            0.1878361089654305191367891 + x * (-2.04233888459752770494024487 + 3.33333333333333333333333333 * x)};
  if constexpr (n == 4)
    return {
        1.5267881254572667869843283 +
            x * (-8.546023607872198765973019 + (14.325858354171888152966621 - 7.420540068038946105200642 * x) * x),
        -0.8136324494869272605618981 +
            x * (13.807166925689577066158695 + x * (-31.388222363446060212058231 + 18.795449407555060811261716 * x)),
        0.400761520311650404800281777 + x * (-7.41707042146263907582738061 +
                                             (24.9981258592191222217269164 - 18.79544940755506081126171563 * x) * x),
        -0.113917196281989931222711973 +
            x * (2.15592710364526077564170438 + x * (-7.9357618499449501626353065 + 7.4205400680389461052006424 * x))};
  return tensor<T, n>{};
}

/**
 * @brief Derivatives of the Lagrange Interpolating polynomials for nodes at Gauss-Legendre points on the interval [-1,
 * 1]
 *
 * Mathematica/Wolfram Language code to generate more entries in the table:
 * Do[xi = GaussianQuadratureWeights[n, 0, 1, 30][[All, 1]];
 *  Print["if constexpr (n == " <> ToString[n] <> ") return " <> ToString[CForm /@ HornerForm /@
 * (D[InterpolatingPolynomial[Transpose[{xi, #}], x], x] & /@ IdentityMatrix[n])] <> ";"], {n, 1, 4}
 * ]
 *
 * @tparam n how many entries to compute
 * @param[in] x where to evaluate the polynomials
 */
template <int n, typename T>
tensor<T, n> GaussLegendreInterpolationDerivative([[maybe_unused]] T x)
{
  if constexpr (n == 1) return {0};
  if constexpr (n == 2) return {-1.7320508075688772935274463415, 1.7320508075688772935274463415};
  if constexpr (n == 3)
    return {-4.6243277820691389617264218 + 6.6666666666666666666666667 * x,
            6.66666666666666666666666667 - 13.3333333333333333333333333 * x,
            -2.04233888459752770494024487 + 6.66666666666666666666666667 * x};
  if constexpr (n == 4)
    return {-8.5460236078721987659730185 + (28.651716708343776305933241 - 22.261620204116838315601927 * x) * x,
            13.8071669256895770661586947 + x * (-62.776444726892120424116461 + 56.386348222665182433785147 * x),
            -7.4170704214626390758273806 + (49.996251718438244443453833 - 56.386348222665182433785147 * x) * x,
            2.15592710364526077564170438 + x * (-15.871523699889900325270613 + 22.2616202041168383156019272 * x)};
  return tensor<T, n>{};
}

}  // namespace serac

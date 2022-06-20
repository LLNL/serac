// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include "serac/numerics/functional/tensor.hpp"

using namespace serac;

static constexpr double tolerance = 4.0e-16;
static constexpr auto   I         = Identity<3>();

template <typename T, int n>
tensor<T, n, n> composeMatrixFromLU(const tensor<int, n>& P, const tensor<T, n, n>& L, const tensor<T, n, n>& U)
{
  auto            LU = dot(L, U);
  tensor<T, n, n> PLU{};
  for (int i = 0; i < n; i++) {
    PLU[P[i]] = LU[i];
  }
  return PLU;
}

TEST(Tensor, BasicOperations)
{
  auto abs = [](auto x) { return (x < 0) ? -x : x; };

  tensor<double, 3> u = {1, 2, 3};
  tensor<double, 4> v = {4, 5, 6, 7};

  tensor<double, 3, 3> A = make_tensor<3, 3>([](int i, int j) { return i + 2.0 * j; });

  double squared_normA = 111.0;
  EXPECT_LT(abs(squared_norm(A) - squared_normA), tolerance);

  tensor<double, 3, 3> symA = {{{0, 1.5, 3}, {1.5, 3, 4.5}, {3, 4.5, 6}}};
  EXPECT_LT(abs(squared_norm(sym(A) - symA)), tolerance);

  tensor<double, 3, 3> devA = {{{-3, 2, 4}, {1, 0, 5}, {2, 4, 3}}};
  EXPECT_LT(abs(squared_norm(dev(A) - devA)), tolerance);

  tensor<double, 3, 3> invAp1 = {{{-4, -1, 3}, {-1.5, 0.5, 0.5}, {2, 0, -1}}};
  EXPECT_LT(abs(squared_norm(inv(A + I) - invAp1)), tolerance);

  tensor<double, 3> Au = {16, 22, 28};
  EXPECT_LT(abs(squared_norm(dot(A, u) - Au)), tolerance);

  tensor<double, 3> uA = {8, 20, 32};
  EXPECT_LT(abs(squared_norm(dot(u, A) - uA)), tolerance);

  double uAu = 144;
  EXPECT_LT(abs(dot(u, A, u) - uAu), tolerance);

  tensor<double, 3, 4> B = make_tensor<3, 4>([](auto i, auto j) { return 3.0 * i - j; });

  double uBv = 300;
  EXPECT_LT(abs(dot(u, B, v) - uBv), tolerance);
}

TEST(Tensor, Elasticity)
{
  static auto abs = [](auto x) { return (x < 0) ? -x : x; };

  double lambda = 5.0;
  double mu     = 3.0;
  tensor C      = make_tensor<3, 3, 3, 3>([&](int i, int j, int k, int l) {
    return lambda * (i == j) * (k == l) + mu * ((i == k) * (j == l) + (i == l) * (j == k));
  });

  auto sigma = [=](auto epsilon) { return lambda * tr(epsilon) * I + 2.0 * mu * epsilon; };

  tensor grad_u = make_tensor<3, 3>([](int i, int j) { return i + 2.0 * j; });

  EXPECT_LT(abs(squared_norm(double_dot(C, sym(grad_u)) - sigma(sym(grad_u)))), tolerance);

  auto epsilon = sym(make_dual(grad_u));

  tensor dsigma_depsilon = get_gradient(sigma(epsilon));

  EXPECT_LT(abs(squared_norm(dsigma_depsilon - C)), tolerance);
}

TEST(Tensor, NavierStokes)
{
  static auto abs = [](auto x) { return (x < 0) ? -x : x; };

  static constexpr double rho   = 3.0;
  static constexpr double mu    = 2.0;
  auto                    sigma = [&](auto p, auto v, auto L) { return rho * outer(v, v) + 2.0 * mu * sym(L) - p * I; };

  auto dsigma_dp = [&](auto /*p*/, auto /*v*/, auto /*L*/) { return -1.0 * I; };

  auto dsigma_dv = [&](auto /*p*/, auto v, auto /*L*/) {
    return make_tensor<3, 3, 3>([&](int i, int j, int k) { return rho * ((i == k) * v[j] + (j == k) * v[i]); });
  };

  auto dsigma_dL = [&](auto /*p*/, auto /*v*/, auto /*L*/) {
    return make_tensor<3, 3, 3, 3>(
        [&](int i, int j, int k, int l) { return mu * ((i == k) * (j == l) + (i == l) * (j == k)); });
  };

  double               p = 3.14;
  tensor               v = {{1.0, 2.0, 3.0}};
  tensor<double, 3, 3> L = {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}};

  {
    auto exact = dsigma_dp(p, v, L);
    auto ad    = get_gradient(sigma(make_dual(p), v, L));
    EXPECT_LT(abs(squared_norm(exact - ad)), tolerance);
  }

  {
    auto exact = dsigma_dv(p, v, L);
    auto ad    = get_gradient(sigma(p, make_dual(v), L));
    EXPECT_LT(abs(squared_norm(exact - ad)), tolerance);
  }

  {
    auto exact = dsigma_dL(p, v, L);
    auto ad    = get_gradient(sigma(p, v, make_dual(L)));
    EXPECT_LT(abs(squared_norm(exact - ad)), tolerance);
  }
}

TEST(Tensor, IsotropicOperations)
{
  double lambda = 5.0;
  double mu     = 3.0;

  tensor<double, 3> u = {1, 2, 3};

  tensor<double, 3, 3> A = make_tensor<3, 3>([](int i, int j) { return i + 2.0 * j; });

  EXPECT_LT(abs(squared_norm(dot(I, u) - u)), tolerance);
  EXPECT_LT(abs(squared_norm(dot(u, I) - u)), tolerance);

  EXPECT_LT(abs(squared_norm(dot(I, A) - A)), tolerance);
  EXPECT_LT(abs(squared_norm(dot(A, I) - A)), tolerance);

  EXPECT_LT(double_dot(I, A) - tr(A), tolerance);

  auto sigma = [=](auto epsilon) { return lambda * tr(epsilon) * I + 2.0 * mu * epsilon; };

  isotropic_tensor<double, 3, 3, 3, 3> C{lambda, 2 * mu, 0.0};

  auto strain = sym(A);

  EXPECT_LT(squared_norm(double_dot(C, strain) - sigma(strain)), tolerance);

  EXPECT_LT(det(I) - 1, tolerance);
  EXPECT_LT(tr(I) - 3, tolerance);
  EXPECT_LT(squared_norm(sym(I) - I), tolerance);
}

TEST(Tensor, ImplicitConversion)
{
  tensor<double, 1> A;
  A(0) = 4.5;

  double value = A;
  EXPECT_NEAR(value, A[0], tolerance);
}

TEST(Tensor, Inverse4x4)
{
  const tensor<double, 4, 4> A{{{2, 1, -1, 1}, {-3, -1, 2, 8}, {-2, 4, 2, 6}, {1, 1, 7, 2}}};
  auto                       invA = inv(A);
  EXPECT_LT(squared_norm(dot(A, invA) - Identity<4>()), tolerance);
}

TEST(Tensor, DerivativeOfInverse)
{
  const tensor<double, 4, 4> A{{{2, 1, -1, 1}, {-3, -1, 2, 8}, {-2, 4, 2, 6}, {1, 1, 7, 2}}};
  auto                       invA = inv(make_dual(A));
  EXPECT_LT(squared_norm(dot(A, get_value(invA)) - Identity<4>()), tolerance);
}

template <int n>
void checkLUDecomposition(const tensor<double, n, n>& A)
{
  auto [P, L, U] = factorize_lu(A);

  // check that L is lower triangular and U is upper triangular
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      EXPECT_DOUBLE_EQ(L[i][j], 0);
      EXPECT_DOUBLE_EQ(U[j][i], 0);
    }
  }

  // check L and U are indeed factors of A
  auto                 LU = dot(L, U);
  tensor<double, n, n> PLU{};
  for (int i = 0; i < n; i++) {
    PLU[P[i]] = LU[i];
  }
  EXPECT_LT(squared_norm(A - PLU), tolerance);
}

TEST(Tensor, LuDecomposition2x2)
{
  const tensor<double, 2, 2> A{{{2, 1}, {-3, -1}}};
  checkLUDecomposition(A);
}

TEST(Tensor, LuDecomposition3x3)
{
  const tensor<double, 3, 3> A{{{2, 1, -1}, {-3, -1, 2}, {-2, 4, 2}}};
  checkLUDecomposition(A);
}

TEST(Tensor, LuDecomposition4x4)
{
  const tensor<double, 4, 4> A{{{2, 1, -1, 1}, {-3, -1, 2, 8}, {-2, 4, 2, 6}, {1, 1, 7, 2}}};
  checkLUDecomposition(A);
}

TEST(Tensor, LuDecompositionWorksOnDualNumbers)
{
  const tensor<double, 3, 3> v{{{2, 1, -1}, {-3, -1, 2}, {-2, 4, 2}}};
  const tensor<double, 3, 3> g{{{0.337494265892494, 0.194238454581911, 0.307832573181341},
                                {0.090147365480304, 0.610402517912401, 0.458978918716148},
                                {0.689309323130592, 0.198321409053159, 0.901973313462065}}};
  tensor<dual<double>, 3, 3> A{};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      A[i][j].value    = v[i][j];
      A[i][j].gradient = g[i][j];
    }
  }
  auto [P, L, U] = factorize_lu(A);
  auto PLU       = composeMatrixFromLU(P, L, U);

  EXPECT_LT(squared_norm(get_value(A) - get_value(PLU)), tolerance);
  EXPECT_LT(squared_norm(get_gradient(A) - get_gradient(PLU)), tolerance);
}

TEST(Tensor, LinearSolveWithOneRhs)
{
  const tensor<double, 3, 3> A{{{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}}};
  const tensor<double, 3>    b{{-1, 2, 3}};

  auto x = linear_solve(A, b);
  EXPECT_LT(squared_norm(dot(A, x) - b), tolerance);
}

TEST(Tensor, LinearSolveWithMultipleRhs)
{
  const tensor<double, 3, 3> A{{{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}}};
  const tensor<double, 3, 2> B{{{-1, 1}, {2, 1}, {3, -2}}};

  auto X = linear_solve(A, B);
  EXPECT_LT(squared_norm(dot(A, X) - B), tolerance);
}

TEST(Tensor, LinearSolveIsConstexprCorrect)
{
  constexpr tensor<double, 3, 3> A{{{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}}};
  constexpr tensor<double, 3>    b{{-1, 2, 3}};
  constexpr auto                 x = linear_solve(A, b);
  EXPECT_LT(squared_norm(dot(A, x) - b), tolerance);
}

TEST(Tensor, DerivativeOfLinearSolve)
{
  // x defined by: t^2 * A * x(t) = t*b
  // implicit derivative 2*t * A * x + t^2 * A * dxdt = b
  // t^2 * A * dxdt = b - 2*t*A*x
  // t^2 * A * dxdt = b - 2*t*b
  // t = 1 --> dxdt = -x

  const tensor<double, 3, 3> A{{{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}}};
  const tensor<double, 3>    b{{-1, 2, 3}};

  auto f = [&A, &b](dual<double> t) { return linear_solve(t * t * A, t * b); };

  double t = 1.0;
  auto   x = f(make_dual(t));

  // expect x_dot = -x
  EXPECT_LT(squared_norm(get_value(x) + get_gradient(x)), tolerance);
}

TEST(Tensor, DerivativeOfLinearSolveWrtBMatchesFiniteDifference)
{
  const tensor<double, 3, 3> A{{{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}}};
  tensor<double, 3>          b_value{{-1, 2, 3}};
  const tensor<double, 3>    b_gradient{{0.337494265892494, 0.194238454581911, 0.307832573181341}};
  auto                       b = make_dual(b_value, b_gradient);

  auto x_value = linear_solve(A, b_value);

  // first order forward difference
  const double h     = 1e-6;
  auto         dx_FD = (linear_solve(A, b_value + h * b_gradient) - x_value) / h;

  auto x = linear_solve(A, b);

  EXPECT_LT(squared_norm(dx_FD - get_gradient(x)), tolerance);
}

TEST(Tensor, DerivativeOfLinearSolveWrtAMatchesFiniteDifference)
{
  const tensor<double, 3, 3> v{{{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}}};
  const tensor<double, 3, 3> g{{{0.337494265892494, 0.194238454581911, 0.307832573181341},
                                {0.090147365480304, 0.610402517912401, 0.458978918716148},
                                {0.689309323130592, 0.198321409053159, 0.901973313462065}}};
  tensor<dual<double>, 3, 3> A{};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      A[i][j].value    = v[i][j];
      A[i][j].gradient = g[i][j];
    }
  }
  const tensor<double, 3> b{{-1, 2, 3}};

  // central difference (2nd order accurate)
  const double h     = 1e-6;
  auto         dx_FD = (linear_solve(v + h * g, b) - linear_solve(v - h * g, b)) / (2 * h);

  auto x = linear_solve(A, b);

  EXPECT_LT(squared_norm(dx_FD - get_gradient(x)), tolerance);
}

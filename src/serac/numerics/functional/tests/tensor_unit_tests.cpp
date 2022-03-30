// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/numerics/functional/tensor.hpp"

#include <gtest/gtest.h>

using namespace serac;

static constexpr double tolerance = 4.0e-16;
static constexpr auto   I         = Identity<3>();

TEST(tensor, basic_operations)
{
  auto abs = [](auto x) { return (x < 0) ? -x : x; };

  tensor<double, 3> u = {1, 2, 3};
  tensor<double, 4> v = {4, 5, 6, 7};

  tensor<double, 3, 3> A = make_tensor<3, 3>([](int i, int j) { return i + 2.0 * j; });

  double sqnormA = 111.0;
  EXPECT_LT(abs(sqnorm(A) - sqnormA), tolerance);

  tensor<double, 3, 3> symA = {{{0, 1.5, 3}, {1.5, 3, 4.5}, {3, 4.5, 6}}};
  EXPECT_LT(abs(sqnorm(sym(A) - symA)), tolerance);

  tensor<double, 3, 3> devA = {{{-3, 2, 4}, {1, 0, 5}, {2, 4, 3}}};
  EXPECT_LT(abs(sqnorm(dev(A) - devA)), tolerance);

  tensor<double, 3, 3> invAp1 = {{{-4, -1, 3}, {-1.5, 0.5, 0.5}, {2, 0, -1}}};
  EXPECT_LT(abs(sqnorm(inv(A + I) - invAp1)), tolerance);

  tensor<double, 3> Au = {16, 22, 28};
  EXPECT_LT(abs(sqnorm(dot(A, u) - Au)), tolerance);

  tensor<double, 3> uA = {8, 20, 32};
  EXPECT_LT(abs(sqnorm(dot(u, A) - uA)), tolerance);

  double uAu = 144;
  EXPECT_LT(abs(dot(u, A, u) - uAu), tolerance);

  tensor<double, 3, 4> B = make_tensor<3, 4>([](auto i, auto j) { return 3.0 * i - j; });

  double uBv = 300;
  EXPECT_LT(abs(dot(u, B, v) - uBv), tolerance);
}

TEST(tensor, elasticity)
{
  static auto abs = [](auto x) { return (x < 0) ? -x : x; };

  double lambda = 5.0;
  double mu     = 3.0;
  tensor C      = make_tensor<3, 3, 3, 3>([&](int i, int j, int k, int l) {
    return lambda * (i == j) * (k == l) + mu * ((i == k) * (j == l) + (i == l) * (j == k));
  });

  auto sigma = [=](auto epsilon) { return lambda * tr(epsilon) * I + 2.0 * mu * epsilon; };

  tensor grad_u = make_tensor<3, 3>([](int i, int j) { return i + 2.0 * j; });

  EXPECT_LT(abs(sqnorm(ddot(C, sym(grad_u)) - sigma(sym(grad_u)))), tolerance);

  auto epsilon = sym(make_dual(grad_u));

  tensor dsigma_depsilon = get_gradient(sigma(epsilon));

  EXPECT_LT(abs(sqnorm(dsigma_depsilon - C)), tolerance);
}

TEST(tensor, navier_stokes)
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
    EXPECT_LT(abs(sqnorm(exact - ad)), tolerance);
  }

  {
    auto exact = dsigma_dv(p, v, L);
    auto ad    = get_gradient(sigma(p, make_dual(v), L));
    EXPECT_LT(abs(sqnorm(exact - ad)), tolerance);
  }

  {
    auto exact = dsigma_dL(p, v, L);
    auto ad    = get_gradient(sigma(p, v, make_dual(L)));
    EXPECT_LT(abs(sqnorm(exact - ad)), tolerance);
  }
}

TEST(tensor, isotropic_operations)
{
  double lambda = 5.0;
  double mu     = 3.0;

  tensor<double, 3> u = {1, 2, 3};

  tensor<double, 3, 3> A = make_tensor<3, 3>([](int i, int j) { return i + 2.0 * j; });

  EXPECT_LT(abs(sqnorm(dot(I, u) - u)), tolerance);
  EXPECT_LT(abs(sqnorm(dot(u, I) - u)), tolerance);

  EXPECT_LT(abs(sqnorm(dot(I, A) - A)), tolerance);
  EXPECT_LT(abs(sqnorm(dot(A, I) - A)), tolerance);

  EXPECT_LT(ddot(I, A) - tr(A), tolerance);

  auto sigma = [=](auto epsilon) { return lambda * tr(epsilon) * I + 2.0 * mu * epsilon; };

  isotropic_tensor<double, 3, 3, 3, 3> C{lambda, 2 * mu, 0.0};

  auto strain = sym(A);

  EXPECT_LT(sqnorm(ddot(C, strain) - sigma(strain)), tolerance);

  EXPECT_LT(det(I) - 1, tolerance);
  EXPECT_LT(tr(I) - 3, tolerance);
  EXPECT_LT(sqnorm(sym(I) - I), tolerance);
}

TEST(tensor, implicit_conversion)
{
  tensor<double, 1> A;
  A(0) = 4.5;

  double value = A;
  EXPECT_NEAR(value, A[0], tolerance);
}

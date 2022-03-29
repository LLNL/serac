// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/numerics/functional/tuple.hpp"
#include "serac/numerics/functional/tuple_arithmetic.hpp"
#include "serac/numerics/functional/tensor.hpp"

#include <random>
#include <iostream>

#include <gtest/gtest.h>

using namespace serac;

auto random_real = [](auto...) {
  static std::default_random_engine             generator;
  static std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  return distribution(generator);
};

static constexpr auto   I   = Identity<3>();
static constexpr double rho = 3.0;
static constexpr double mu  = 2.0;

static constexpr double epsilon = 1.0e-6;

TEST(TupleArithmeticUnitTests, structured_binding)
{
  serac::tuple x{0, 1.0, 2.0f};
  auto [a, b, c] = x;
  EXPECT_NEAR(a, 0, 1.0e-10);
  EXPECT_NEAR(b, 1.00, 1.0e-10);
  EXPECT_NEAR(c, 2.0f, 1.0e-10);
}

TEST(TupleArithmeticUnitTests, add)
{
  serac::tuple a{0.0, make_tensor<3>([](int) { return 3.0; }),
                 make_tensor<5, 3>([](int i, int j) { return 1.0 / (i + j + 1); })};
  serac::tuple b = a + a;
  EXPECT_NEAR(serac::get<0>(b), 0.0, 1.0e-10);
  EXPECT_NEAR(norm(serac::get<1>(b)), 10.39230484541326, 1.0e-10);
  EXPECT_NEAR(norm(serac::get<2>(b)), 2.977782431376876, 1.0e-10);
}

TEST(TupleArithmeticUnitTests, subtract)
{
  serac::tuple a{0.0, make_tensor<3>([](int) { return 3.0; }),
                 make_tensor<5, 3>([](int i, int j) { return 1.0 / (i + j + 1); })};
  serac::tuple b = a - a;
  EXPECT_NEAR(serac::get<0>(b), 0.0, 1.0e-10);
  EXPECT_NEAR(norm(serac::get<1>(b)), 0.0, 1.0e-10);
  EXPECT_NEAR(norm(serac::get<2>(b)), 0.0, 1.0e-10);
}

TEST(TupleArithmeticUnitTests, multiply)
{
  serac::tuple a{0.0, make_tensor<3>([](int) { return 3.0; }),
                 make_tensor<5, 3>([](int i, int j) { return 1.0 / (i + j + 1); })};
  serac::tuple b = 2.0 * a;
  EXPECT_NEAR(serac::get<0>(b), 0.0, 1.0e-10);
  EXPECT_NEAR(norm(serac::get<1>(b)), 10.39230484541326, 1.0e-10);
  EXPECT_NEAR(norm(serac::get<2>(b)), 2.977782431376876, 1.0e-10);
}

TEST(TupleArithmeticUnitTests, divide)
{
  serac::tuple a{0.0, make_tensor<3>([](int) { return 3.0; }),
                 make_tensor<5, 3>([](int i, int j) { return 1.0 / (i + j + 1); })};
  serac::tuple b = a / 0.5;
  EXPECT_NEAR(serac::get<0>(b), 0.0, 1.0e-10);
  EXPECT_NEAR(norm(serac::get<1>(b)), 10.39230484541326, 1.0e-10);
  EXPECT_NEAR(norm(serac::get<2>(b)), 2.977782431376876, 1.0e-10);
}

TEST(TupleArithmeticUnitTests, tensor_output_with_tuple_input)
{
  constexpr auto f = [=](auto p, auto v, auto L) { return rho * outer(v, v) * det(I + L) + 2.0 * mu * sym(L) - p * I; };

  [[maybe_unused]] constexpr double p = 3.14;
  [[maybe_unused]] constexpr tensor v = {{1.0, 2.0, 3.0}};
  constexpr tensor<double, 3, 3>    L = {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}};

  constexpr double               dp = 1.23;
  constexpr tensor               dv = {{2.0, 1.0, 4.0}};
  constexpr tensor<double, 3, 3> dL = {{{3.0, 1.0, 2.0}, {2.0, 7.0, 3.0}, {4.0, 4.0, 3.0}}};

  auto dfdp = get_gradient(f(make_dual(p), v, L));
  auto dfdv = get_gradient(f(p, make_dual(v), L));
  auto dfdL = get_gradient(f(p, v, make_dual(L)));

  auto df0 = (f(p + epsilon * dp, v + epsilon * dv, L + epsilon * dL) -
              f(p - epsilon * dp, v - epsilon * dv, L - epsilon * dL)) /
             (2 * epsilon);

  auto df1 = dfdp * dp + dfdv * dv + ddot(dfdL, dL);

  EXPECT_NEAR(norm(df1 - df0) / norm(df0), 0.0, 2.0e-8);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

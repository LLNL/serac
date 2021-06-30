// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "../tuple_arithmetic.hpp"

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

TEST(TupleTests, add)
{
  std::tuple a{0.0, make_tensor<3>([](int) { return 3.0; }),
               make_tensor<5, 3>([](int i, int j) { return 1.0 / (i + j + 1); })};
  std::tuple b = a + a;
  EXPECT_NEAR(std::get<0>(b), 0.0, 1.0e-10);
  EXPECT_NEAR(norm(std::get<1>(b)), 10.39230484541326, 1.0e-10);
  EXPECT_NEAR(norm(std::get<2>(b)), 2.977782431376876, 1.0e-10);
}

TEST(TupleTests, subtract)
{
  std::tuple a{0.0, make_tensor<3>([](int) { return 3.0; }),
               make_tensor<5, 3>([](int i, int j) { return 1.0 / (i + j + 1); })};
  std::tuple b = a - a;
  EXPECT_NEAR(std::get<0>(b), 0.0, 1.0e-10);
  EXPECT_NEAR(norm(std::get<1>(b)), 0.0, 1.0e-10);
  EXPECT_NEAR(norm(std::get<2>(b)), 0.0, 1.0e-10);
}

TEST(TupleTests, multiply)
{
  std::tuple a{0.0, make_tensor<3>([](int) { return 3.0; }),
               make_tensor<5, 3>([](int i, int j) { return 1.0 / (i + j + 1); })};
  std::tuple b = 2.0 * a;
  EXPECT_NEAR(std::get<0>(b), 0.0, 1.0e-10);
  EXPECT_NEAR(norm(std::get<1>(b)), 10.39230484541326, 1.0e-10);
  EXPECT_NEAR(norm(std::get<2>(b)), 2.977782431376876, 1.0e-10);
}

TEST(TupleTests, divide)
{
  std::tuple a{0.0, make_tensor<3>([](int) { return 3.0; }),
               make_tensor<5, 3>([](int i, int j) { return 1.0 / (i + j + 1); })};
  std::tuple b = a / 0.5;
  EXPECT_NEAR(std::get<0>(b), 0.0, 1.0e-10);
  EXPECT_NEAR(norm(std::get<1>(b)), 10.39230484541326, 1.0e-10);
  EXPECT_NEAR(norm(std::get<2>(b)), 2.977782431376876, 1.0e-10);
}

TEST(ChainRuleTests, scalar_output_with_scalar_input)
{
  auto f = [](auto x) { return x * x + 4.0; };

  double x  = 1.36;
  double dx = 1.0;

  auto output = f(make_dual(x));

  auto fx   = get_value(output);
  auto dfdx = get_gradient(output);

  EXPECT_NEAR(f(x), fx, 1.e-13);
  EXPECT_NEAR((f(x + epsilon * dx) - f(x)) / epsilon, chain_rule(dfdx, dx), 1.e-6);
}

TEST(ChainRuleTests, vector_output_with_vector_input)
{
  auto f = [](auto x) {
    auto tmp = norm(x) * x;
    tmp[0] -= 3.0;
    return tmp;
  };

  auto x  = make_tensor<3>(random_real);
  auto dx = make_tensor<3>(random_real);

  auto output = f(make_dual(x));

  auto value = get_value(output);
  auto dfdx  = get_gradient(output);

  EXPECT_NEAR(norm(f(x) - value), 0.0, 1.e-13);
  EXPECT_NEAR(norm(((f(x + epsilon * dx) - f(x)) / epsilon) - chain_rule(dfdx, dx)), 0.0, 1.e-6);
}

TEST(ChainRuleTests, matrix_output_with_matrix_input)
{
  auto f = [](auto x) { return inv(x + I) - x; };

  auto x  = make_tensor<3, 3>(random_real);
  auto dx = make_tensor<3, 3>(random_real);

  auto output = f(make_dual(x));

  auto value = get_value(output);
  auto dfdx  = get_gradient(output);

  EXPECT_NEAR(norm(f(x) - value), 0.0, 1.e-13);
  EXPECT_NEAR(norm(((f(x + epsilon * dx) - f(x - epsilon * dx)) / (2 * epsilon)) - chain_rule(dfdx, dx)), 0.0, 1.e-8);
}

TEST(ChainRuleTests, scalar_output_with_matrix_input)
{
  auto f = [](auto x) { return tr(x) * det(x); };

  auto x  = make_tensor<3, 3>(random_real);
  auto dx = make_tensor<3, 3>(random_real);

  auto output = f(make_dual(x));

  auto value = get_value(output);
  auto dfdx  = get_gradient(output);

  EXPECT_NEAR(f(x) - value, 0.0, 1.e-13);
  EXPECT_NEAR(((f(x + epsilon * dx) - f(x)) / epsilon) - chain_rule(dfdx, dx), 0.0, 1.e-6);
}

TEST(ChainRuleTests, tuple_output_with_tuple_input)
{
  constexpr auto f = [](auto p, auto v, auto L) {
    return std::tuple{rho * outer(v, v) + 2.0 * mu * sym(L) - p * I, v + dot(v, L)};
  };

  [[maybe_unused]] constexpr double p = 3.14;
  [[maybe_unused]] constexpr tensor v = {{1.0, 2.0, 3.0}};
  // CUDA workaround template deduction guide failed
  constexpr tensor<double, 3, 3> L = {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}};

  constexpr double dp = 1.23;
  constexpr tensor dv = {{2.0, 1.0, 4.0}};
  // CUDA workaround template deduction guide failed
  constexpr tensor<double, 3, 3> dL = {{{3.0, 1.0, 2.0}, {2.0, 7.0, 3.0}, {4.0, 4.0, 3.0}}};

  auto output = std::apply(f, make_dual(p, v, L));

  auto value = get_value(output);
  auto grad  = get_gradient(output);

  auto df_fd = (f(p + epsilon * dp, v + epsilon * dv, L + epsilon * dL) -
                f(p - epsilon * dp, v - epsilon * dv, L - epsilon * dL)) /
               (2 * epsilon);

  auto df0 = (std::get<0>(std::get<0>(grad)) * dp) + dot(std::get<1>(std::get<0>(grad)), dv) +
             ddot(std::get<2>(std::get<0>(grad)), dL);

  auto df1 = (std::get<0>(std::get<1>(grad)) * dp) + dot(std::get<1>(std::get<1>(grad)), dv) +
             ddot(std::get<2>(std::get<1>(grad)), dL);

  auto df_ad = chain_rule(grad, std::tuple{dp, dv, dL});

  EXPECT_NEAR(norm(std::get<0>(f(p, v, L)) - std::get<0>(value)), 0.0, 1.e-13);
  EXPECT_NEAR(norm(std::get<1>(f(p, v, L)) - std::get<1>(value)), 0.0, 1.e-13);

  EXPECT_NEAR(norm(std::get<0>(df_ad) - df0), 0.0, 1.e-8);
  EXPECT_NEAR(norm(std::get<1>(df_ad) - df1), 0.0, 1.e-8);
  EXPECT_NEAR(norm(std::get<0>(df_ad) - std::get<0>(df_fd)), 0.0, 1.e-8);
  EXPECT_NEAR(norm(std::get<1>(df_ad) - std::get<1>(df_fd)), 0.0, 1.e-8);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

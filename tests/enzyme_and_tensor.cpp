#include "gtest/gtest.h"

#include <stdio.h>

#include "tuple.hpp"
#include "tensor.hpp"

#include "jax_wrapper.hpp"

using namespace serac;

double foo(tensor<double, 3> x) {
  return dot(x, x);
}

double dfoo(tensor<double, 3> x, tensor< double, 3 > dx) {
  return __enzyme_fwddiff<double>((void*) foo, x, dx);
}

TEST(enzyme, basic_test) {
  for(double i=1; i<5; i++) {
    tensor<double, 3> x{i, i, i};
    tensor<double, 3> dx{1.0, 0.0, 0.0};
    EXPECT_EQ(foo(x), 3*i*i);
    EXPECT_EQ(dfoo(x, dx), 2*i);
  }
}

tensor<double,3> bar(tensor<double, 3, 3> A, tensor<double, 3> x) {
  return dot(inv(A), x);
}

tensor<double,3> dbar(tensor< double, 3, 3 > A, tensor< double, 3, 3 > dA, tensor<double, 3> x) {
  return __enzyme_fwddiff<tensor<double,3>>((void*) bar, enzyme_dup, A, dA, enzyme_const, x);
}

TEST(enzyme, linear_solve_test) {
  tensor<double,3,3> A = {{
    {2, 1, 0},
    {1, 2, 1},
    {0, 1, 2}
  }};
  tensor<double,3,3> dA = {{
    {1, 0, 1},
    {1, 0, 0},
    {0, 1, 0}
  }};
  tensor<double, 3> x{1.0, 2.0, 3.0};

  tensor<double, 3> y = bar(A, x);
  EXPECT_EQ(y[0], 0.5);
  EXPECT_EQ(y[1], 0.0);
  EXPECT_EQ(y[2], 1.5);

  tensor<double, 3> dy = dbar(A, dA, x);
  EXPECT_EQ(dy[0], -1.25);
  EXPECT_EQ(dy[1],  0.50);
  EXPECT_EQ(dy[2], -0.25);
}

double square(double x) { return x * x; }

TEST(enzyme, jvp_test_1_arg) {
  double x = 3.0;
  double dx = 1.5;

  auto square_jvp = jvp<square>(x);
  EXPECT_EQ(square_jvp(dx), 9.0);
}

TEST(enzyme, jvp_test_2_args) {
  tensor<double,3,3> A = {{
    {2, 1, 0},
    {1, 2, 1},
    {0, 1, 2}
  }};
  tensor<double,3,3> dA = {{
    {1, 0, 1},
    {1, 0, 0},
    {0, 1, 0}
  }};
  tensor<double, 3> x{1.0, 2.0, 3.0};

  tensor<double, 3> y = bar(A, x);
  EXPECT_EQ(y[0], 0.5);
  EXPECT_EQ(y[1], 0.0);
  EXPECT_EQ(y[2], 1.5);

  tensor<double, 3> dx{};
  tensor<double, 3> dy = (jvp<bar>(A, x))(dA, dx);
  EXPECT_EQ(dy[0], -1.25);
  EXPECT_EQ(dy[1],  0.50);
  EXPECT_EQ(dy[2], -0.25);
}

TEST(enzyme, vjp_test_1_arg) {

  double x = 3.5;
  auto f_vjp = vjp<square>(x);

  double y = 1.7;
  double z = f_vjp(1.7);

  double dfdx = x * 2.0;
  EXPECT_NEAR(z, dfdx * y, 1.0e-15);

}

TEST(enzyme, vjp_test_2_args) {

  // note: the unary "+" operator converts 
  // a stateless lambda function to a function pointer
  constexpr auto f = +[](double x, double y) {
    return std::tuple{sin(x), cos(y)};
  };

  auto f_vjp = vjp<f>(0.5, 1.0);

  auto [xbar, ybar] = f_vjp(std::tuple{-0.7, 0.3});

  EXPECT_NEAR(xbar, -0.6143077933232609, 1.0e-15);
  EXPECT_NEAR(ybar, -0.2524412954423689, 1.0e-15);

}

TEST(enzyme, vjp_test_3_args) {

  // note: the unary "+" operator converts 
  // a stateless lambda function to a function pointer
  constexpr auto f = +[](double x, double y, double z) {
    return std::tuple{sin(x), cos(y), tan(z)};
  };

  auto f_vjp = vjp<f>(0.5, 1.0, 2.0);

  auto [xbar, ybar, zbar] = f_vjp(std::tuple{-0.7, 0.3, -0.1});

  EXPECT_NEAR(xbar, -0.6143077933232609, 1.0e-15);
  EXPECT_NEAR(ybar, -0.2524412954423689, 1.0e-15);
  EXPECT_NEAR(zbar, -0.5774399204041918, 1.0e-15);

}
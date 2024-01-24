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

void sqr(double x, double& out)
{
  out = x*x;
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

#if 0
void another_linear_solve(tensor<double, 3, 3> A, tensor<double, 3> b, tensor<double, 3>& x)
{
  x = linear_solve(A, b);
}

void another_linear_solve_fwddiff(const tensor<double, 3, 3>* A, const tensor<double, 3, 3>* dA,
                                  const tensor<double, 3>* b, const tensor<double, 3>* db,
                                  tensor<double, 3>* x, tensor<double, 3>* dx)

{
  *x = linear_solve(*A, *b);
  tensor<double, 3> r = *db - dot(*dA, *x);
  *dx = linear_solve(*A, r);
}

void* __enzyme_register_derivative_another_linear_solve[] = {(void*) another_linear_solve, (void*) another_linear_solve_fwddiff};

TEST(enzyme, another_linear_solve_test) {
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
  tensor<double, 3> b{1.0, 2.0, 3.0};
  tensor<double, 3> db{};

  tensor<double, 3> x{};

  const double TOL = 1e-15;

  another_linear_solve(A, b, x);
  EXPECT_NEAR(x[0], 0.5, TOL);
  EXPECT_NEAR(x[1], 0.0, TOL);
  EXPECT_NEAR(x[2], 1.5, TOL);

  tensor<double, 3> dx{};

  __enzyme_fwddiff<void>((void*) another_linear_solve, &A, &dA, &b, &db, &x, &dx);
  EXPECT_NEAR(dx[0], -1.25, TOL);
  EXPECT_NEAR(dx[1],  0.50, TOL);
  EXPECT_NEAR(dx[2], -0.25, TOL);
}
#endif

TEST(enzyme, fwddiffOnVoidFn)
{
  double x = 2.0;

  double y = 0;
  sqr(x, y);
  EXPECT_DOUBLE_EQ(y, 4.0);

  double dx = 1.0;
  double dydx = 0;
  y = 0.0;
  __enzyme_fwddiff<void>((void*) sqr, enzyme_dup, x, dx, enzyme_dup, &y, &dydx);
  EXPECT_DOUBLE_EQ(y, 4.0);
  EXPECT_DOUBLE_EQ(dydx, 4.0);
}

double square(double x) { return x * x; }

TEST(enzyme, jvp_test_1_arg) {
  double x = 3.0;
  double dx = 1.5;

  auto square_jvp = jvp<square>(x);
  EXPECT_EQ(square_jvp(dx), 9.0);
}

TEST(enzyme, vjp_test_1_arg) {

  double x = 3.5;
  auto f_vjp = vjp<square>(x);

  double y = 1.7;
  double z = f_vjp(1.7);

  double dfdx = x * 2.0;
  EXPECT_NEAR(z, dfdx * y, 1.0e-15);

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

template < typename T, typename ... arg_types >
auto functor_wrapper(const T & f, arg_types && ... args) {
  return f(args ...);
}

struct Functor {
  auto operator() (const std::array<double,2> & x) const {
    return x[0] * a + x[1] * x[1] * b * b;
  }

  double a;
  double b;
};

TEST(enzyme, manual_jvp_test_functor) {
  Functor f{1.0, 2.0};

  std::array<double, 2> x{3.0, 4.0};
  std::array<double, 2> dx{5.0, 6.0};
  double y = f({3.0, 4.0});
  EXPECT_EQ(y, 67.0 /* 3 * 1 + (4 * 4) * (2 * 2) */);

  double dy = __enzyme_fwddiff<double>((void*)functor_wrapper<decltype(f), std::array<double,2> >, enzyme_const, (void*)&f, enzyme_dup, &x, &dx);
  EXPECT_EQ(dy, 197.0);
}

TEST(enzyme, manual_jvp_test_lambda) {
  auto f = [a = 1.0, b = 2.0](std::array<double,2> x) {
    return x[0] * a + x[1] * x[1] * b * b;
  };

  std::array<double, 2> x{3.0, 4.0};
  std::array<double, 2> dx{5.0, 6.0};
  double y = f({3.0, 4.0});
  EXPECT_EQ(y, 67.0 /* 3 * 1 + (4 * 4) * (2 * 2) */);

  double dy = __enzyme_fwddiff<double>((void*)functor_wrapper<decltype(f), std::array<double,2> >, enzyme_const, (void*)&f, enzyme_dup, &x, &dx);
  EXPECT_EQ(dy, 197.0);
}


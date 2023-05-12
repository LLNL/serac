#include "gtest/gtest.h"

#include <stdio.h>

#include "tuple.hpp"
#include "tensor.hpp"

using namespace serac;

int enzyme_dup;
int enzyme_out;
int enzyme_const;

template < typename return_type, typename ... T >
extern return_type __enzyme_fwddiff(void*, T ... );

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

void another_linear_solve(tensor<double, 3, 3> A, tensor<double, 3> b, tensor<double, 3>& x)
{
  x = linear_solve(A, b);
}

void another_linear_solve_fwddiff(tensor<double, 3, 3>* A, tensor<double, 3, 3>* dA, 
                                  tensor<double, 3>* b, tensor<double, 3>* db,
                                  tensor<double, 3>* x, tensor<double, 3>* dx)

{
  std::cout << "inside derivative fn" << std::endl;
  std::cout << "A = " << (*A) << std::endl;
  std::cout << "dA = " << (*dA) << std::endl;
  std::cout << "b = " << (*b) << std::endl;
  std::cout << "db = " << (*db) << std::endl;
  (*x) = linear_solve(*A, *b);
  std::cout << "x = \n" << (*x) << std::endl;
  std::cout << "dx = \n" << (*dx) << std::endl;
  tensor<double, 3> r = (*db) - dot(*dA, *x);
  std::cout << "r = " << r << std::endl;
  (*dx) = linear_solve(*A, r);
}

void* __enzyme_register_derivative_another_linear_solve[] = {(void*) another_linear_solve, (void*) another_linear_solve_fwddiff};


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

  std::cout << "b = " << b << std::endl;
  tensor<double, 3> dx{};
  std::cout << "x = \n" << x << std::endl;
  tensor<double, 3>* px = &x;
  std::cout << "address of x in test = " << px <<std::endl;
  std::cout << "x through indirection, " << (*px) << std::endl;

  __enzyme_fwddiff<void>((void*) another_linear_solve, &A, &dA, &b, &db, &x, &dx);
  std::cout << "back in test" << std::endl;
  std::cout << "x = \n" << x << std::endl;
  std::cout << "dx = \n" << dx << std::endl;
  EXPECT_NEAR(dx[0], -1.25, TOL);
  EXPECT_NEAR(dx[1],  0.50, TOL);
  EXPECT_NEAR(dx[2], -0.25, TOL);
}

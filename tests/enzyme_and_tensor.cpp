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
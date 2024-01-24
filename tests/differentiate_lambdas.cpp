#include "gtest/gtest.h"

#include <stdio.h>

extern double __enzyme_autodiff(void*, double);

double square(double x) {
  return x * x;
}

double dsquare(double x) {
  return __enzyme_autodiff((void*) square, x);
}

TEST(enzyme, basic_test) {
  for(double i=1; i<5; i++) {
    EXPECT_EQ(square(i), i*i);
    EXPECT_EQ(dsquare(i), 2*i);
  }
}
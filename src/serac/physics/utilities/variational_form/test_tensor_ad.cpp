#include "mfem.hpp"
#include "tensor.hpp"

#include <gtest/gtest.h>

using namespace mfem;

TEST(tensor, norm)
{
  tensor<double, 5> a = {{1.0, 2.0, 3.0, 4.0, 5.0}};
  EXPECT_DOUBLE_EQ(norm(a) - sqrt(55), 0.0);
}

const auto   eps = std::numeric_limits<double>::epsilon();
const double x   = 0.5;

TEST(dual_number_tensor, cos)
{
  auto xd = cos(make_dual(x));
  EXPECT_DOUBLE_EQ(abs(-sin(x) - xd.gradient), 0.0);
}

TEST(dual_number_tensor, exp)
{
  auto xd = exp(make_dual(x));
  EXPECT_DOUBLE_EQ(abs(exp(x) - xd.gradient), 0.0);
}

TEST(dual_number_tensor, log)
{
  auto xd = log(make_dual(x));
  EXPECT_DOUBLE_EQ(abs(1.0 / x - xd.gradient), 0.0);
}

TEST(dual_number_tensor, pow)
{
  // f(x) = x^3/2
  auto xd = pow(make_dual(x), 1.5);
  EXPECT_DOUBLE_EQ(abs(1.5 * pow(x, 0.5) - xd.gradient), 0.0);
}

TEST(dual_number_tensor, mixed_operations)
{
  auto xd = make_dual(x);
  auto r  = cos(xd) * cos(xd);
  EXPECT_DOUBLE_EQ(abs(-2.0 * sin(x) * cos(x) - r.gradient), 0.0);

  r = exp(xd) * cos(xd);
  EXPECT_LT(abs(exp(x) * (cos(x) - sin(x)) - r.gradient), eps);

  r = log(xd) * cos(xd);
  EXPECT_LT(abs((cos(x) / x - log(x) * sin(x)) - r.gradient), eps);

  r = exp(xd) * pow(xd, 1.5);
  EXPECT_LT(abs((exp(x) * (pow(x, 1.5) + 1.5 * pow(x, 0.5))) - r.gradient), eps);

  tensor<double, 2> vx  = {{0.5, 0.25}};
  tensor<double, 2> vre = {{0.894427190999916, 0.4472135954999579}};
  auto              vr  = norm(make_dual(vx));
  EXPECT_LT(norm(vr.gradient - vre), eps);
}

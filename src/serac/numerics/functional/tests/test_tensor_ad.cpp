// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/numerics/functional/tensor.hpp"

using namespace mfem;
using namespace serac;

TEST(Tensor, Norm)
{
  tensor<double, 5> a = {{1.0, 2.0, 3.0, 4.0, 5.0}};
  EXPECT_DOUBLE_EQ(norm(a) - sqrt(55), 0.0);
}

const auto   eps = std::numeric_limits<double>::epsilon();
const double x   = 0.5;
const double y   = 0.25;

TEST(DualNumberTensor, Max)
{
  auto xd = max(make_dual(x), make_dual(y));
  EXPECT_DOUBLE_EQ(std::max(x, y), xd.value);
  EXPECT_DOUBLE_EQ(1.0, xd.gradient);

  auto xd2 = max(x, make_dual(y));
  EXPECT_DOUBLE_EQ(std::max(x, y), xd2.value);
  EXPECT_DOUBLE_EQ(0.0, xd2.gradient);

  auto xd3 = max(make_dual(x), y);
  EXPECT_DOUBLE_EQ(std::max(x, y), xd3.value);
  EXPECT_DOUBLE_EQ(1.0, xd3.gradient);
}

TEST(DualNumberTensor, Min)
{
  auto xd = min(make_dual(x), make_dual(y));
  EXPECT_DOUBLE_EQ(std::min(x, y), xd.value);
  EXPECT_DOUBLE_EQ(1.0, xd.gradient);

  auto xd2 = min(x, make_dual(y));
  EXPECT_DOUBLE_EQ(std::min(x, y), xd2.value);
  EXPECT_DOUBLE_EQ(1.0, xd2.gradient);

  auto xd3 = min(make_dual(x), y);
  EXPECT_DOUBLE_EQ(std::min(x, y), xd3.value);
  EXPECT_DOUBLE_EQ(0.0, xd3.gradient);
}

TEST(DualNumberTensor, Cos)
{
  auto xd = cos(make_dual(x));
  EXPECT_DOUBLE_EQ(abs(-sin(x) - xd.gradient), 0.0);
}

TEST(DualNumberTensor, Atan)
{
  auto xd = atan(make_dual(x));
  EXPECT_DOUBLE_EQ(abs(1.0 / (1.0 + pow(x, 2.0)) - xd.gradient), 0.0);
  EXPECT_DOUBLE_EQ(atan(x) - xd.value, 0.0);
}

TEST(DualNumberTensor, Atan2)
{
  auto xd = atan2(make_dual(y), make_dual(x));
  EXPECT_DOUBLE_EQ(abs((x / (pow(x, 2.0) + pow(y, 2.0)) - y / (pow(x, 2.0) + pow(y, 2.0))) - xd.gradient), 0.0);
  EXPECT_DOUBLE_EQ(atan2(y, x) - xd.value, 0.0);

  auto xd1 = atan2(make_dual(y), x);
  EXPECT_DOUBLE_EQ(abs(x / (pow(x, 2.0) + pow(y, 2.0)) - xd1.gradient), 0.0);
  EXPECT_DOUBLE_EQ(atan2(y, x) - xd1.value, 0.0);

  auto xd2 = atan2(y, make_dual(x));
  EXPECT_DOUBLE_EQ(abs(-y / (pow(x, 2.0) + pow(y, 2.0)) - xd2.gradient), 0.0);
  EXPECT_DOUBLE_EQ(atan2(y, x) - xd2.value, 0.0);
}

TEST(DualNumberTensor, Asin)
{
  auto xd = asin(make_dual(x));
  EXPECT_DOUBLE_EQ(abs(1.0 / sqrt(1.0 - pow(x, 2.0)) - xd.gradient), 0.0);
}

TEST(DualNumberTensor, Acos)
{
  auto xd = acos(make_dual(x));
  EXPECT_DOUBLE_EQ(abs(-1.0 / sqrt(1.0 - pow(x, 2.0)) - xd.gradient), 0.0);
}

TEST(DualNumberTensor, Exp)
{
  auto xd = exp(make_dual(x));
  EXPECT_DOUBLE_EQ(abs(exp(x) - xd.gradient), 0.0);
}

TEST(DualNumberTensor, Log)
{
  auto xd = log(make_dual(x));
  EXPECT_DOUBLE_EQ(abs(1.0 / x - xd.gradient), 0.0);
}

TEST(DualNumberTensor, Pow)
{
  // f(x) = x^3/2
  auto xd = pow(make_dual(x), 1.5);
  EXPECT_DOUBLE_EQ(abs(1.5 * pow(x, 0.5) - xd.gradient), 0.0);
}

TEST(DualNumberTensor, MixedOperations)
{
  auto xd = make_dual(x);
  auto r  = cos(xd) * cos(xd);
  EXPECT_DOUBLE_EQ(abs(-2.0 * sin(x) * cos(x) - r.gradient), 0.0);

  r = exp(xd) * cos(xd);
  EXPECT_LT(abs(exp(x) * (cos(x) - sin(x)) - r.gradient), eps);

  r = log(xd) * cos(xd);
  EXPECT_LT(abs((cos(x) / x - std::log(x) * sin(x)) - r.gradient), eps);

  r = exp(xd) * pow(xd, 1.5);
  EXPECT_LT(abs((exp(x) * (pow(x, 1.5) + 1.5 * pow(x, 0.5))) - r.gradient), eps);

  tensor<double, 2> vx  = {{0.5, 0.25}};
  tensor<double, 2> vre = {{0.894427190999916, 0.4472135954999579}};
  auto              vr  = norm(make_dual(vx));
  EXPECT_LT(norm(vr.gradient - vre), eps);
}

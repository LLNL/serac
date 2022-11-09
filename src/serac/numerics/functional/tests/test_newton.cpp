// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include "serac/numerics/functional/tensor.hpp"

using namespace serac;

template <typename T>
T f(T x)
{
    return x*x - 2.0;
}

template <typename T>
T g(T x, double p)
{
    return x*x - p;
}

template <typename T>
T function_that_kills_newton(T x)
{
    return sin(x) + x;
}

TEST(ScalarEquationSolver, Converges)
{
    double x0 = 2.0;
    double tolerance = 1e-8;
    double lower = 1e-3;
    double upper = 2.5;
    double x = solve_scalar_equation([](auto x){ return f(x);}, x0, tolerance, lower, upper);
    double error = std::abs((x - std::sqrt(2.0))/std::sqrt(2.0));
    EXPECT_LT(error, tolerance);
}

TEST(ScalarEquationSolver, WorksWithParameter)
{
    double x0 = 2.0;
    double tolerance = 1e-8;
    double lower = 1e-3;
    double upper = 2.5;
    double p = 2.0;
    double x = solve_scalar_equation([](auto x, auto a){ return g(x, a);}, x0, tolerance, lower, upper, p);
    double error = std::abs((x - std::sqrt(p))/std::sqrt(p));
    EXPECT_LT(error, tolerance);
}

// TEST(ScalarEquationSolver, AbortsIfRootNotBracketedByCaller)
// {
//     double x0 = 5.0;
//     double tolerance = 1e-8;
//     double lower = 2.0;
//     double upper = 10.0;
//     double x = solve_scalar_equation([](auto x){ return f(x);}, x0, tolerance, lower, upper);
//     std::cout << "x = " << x << std::endl;
// }

TEST(ScalarEquationSolver, ConvergesWithGuessOutsideNewtonBasin)
{
    double x0 = 9.5;
    double tolerance = 1e-8;
    double lower = -10.0;
    double upper = 10.0;
    double x = solve_scalar_equation([](auto x) { return function_that_kills_newton(x); }, 
                                     x0, tolerance, lower, upper);
    double error = std::abs(x);
    std::cout << "x = " << x << std::endl;
    EXPECT_LT(error, tolerance);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
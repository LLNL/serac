// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include "serac/numerics/functional/tuple_tensor_dual_functions.hpp"

using namespace serac;

TEST(ScalarEquationSolver, ConvergesOnEasyProblem)
{
    double x0 = 2.0;
    double tolerance = 1e-8;
    double lower = 1e-3;
    double upper = 2.5;
    auto result = solve_scalar_equation([](auto x){ return exp(x) - 2.0; }, x0, tolerance, lower, upper);
    auto x = result.root;
    double exact = std::log(2.0);
    double error = std::abs((x - exact)/abs(exact));
    EXPECT_LT(error, tolerance);
}

TEST(ScalarEquationSolver, WorksWithScalarParameter)
{
    auto my_sqrt = [](dual<double> p) {
        double x0 = get_value(p);
        double tolerance = 1e-6;
        double lower = 0;
        double upper = (get_value(p) > 1.0) ? get_value(p) : 1.0;
        auto result = solve_scalar_equation([](auto x, auto a){ return x*x - a; }, x0, tolerance, lower, upper, p);
        return result.root;
    };
    double p = 2.0;
    auto [sqrt_p, dsqrt_p] = my_sqrt(make_dual(p));

    // check value
    double exact_value = std::sqrt(2.0);
    EXPECT_LT(std::abs(sqrt_p - exact_value)/exact_value, 1e-12);

    double exact_derivative = 0.5/std::sqrt(p);
    EXPECT_LT(std::abs(dsqrt_p - exact_derivative)/std::abs(exact_derivative), 1e-12);
}

TEST(ScalarEquationSolver, AbortsIfRootNotBracketedByCaller)
{
    double x0 = 5.0;
    double tolerance = 1e-8;
    double lower = 2.0;
    double upper = 10.0;
    EXPECT_DEATH_IF_SUPPORTED({
            [[maybe_unused]] auto result = 
                solve_scalar_equation([](auto x){ return x*x - 2.0; }, x0, tolerance, lower, upper);
        }, "solve_scalar_equation: root not bracketed by input bounds.");
}

TEST(ScalarEquationSolver, ReturnsImmediatelyIfUpperBoundIsARoot)
{
    double p = 4.0;
    double upper = 2.0;
    
    double x0 = 1.0;
    double tolerance = 1e-8;
    double lower = 0.0;

    auto result = solve_scalar_equation([](auto x, auto a){ return x*x - a; }, x0, tolerance, lower, upper, p);

    double error = std::abs((result.root - 2.0))/2.0;
    EXPECT_LT(error, tolerance);

    EXPECT_EQ(result.iterations, 0);
}

TEST(ScalarEquationSolver, ReturnsImmediatelyIfLowerBoundIsARoot)
{
    double p = 4.0;
    double lower = 2.0;
    
    double x0 = 6.0;
    double tolerance = 1e-8;
    double upper = 8.0;

    auto result = solve_scalar_equation([](auto x, auto a){ return x*x - a; }, x0, tolerance, lower, upper, p);

    double error = std::abs((result.root - 2.0))/2.0;
    EXPECT_LT(error, tolerance);

    EXPECT_EQ(result.iterations, 0);
}


TEST(ScalarEquationSolver, ConvergesWithGuessOutsideNewtonBasin)
{
    double x0 = 9.5;
    double tolerance = 1e-8;
    double lower = -10.0;
    double upper = 10.0;
    auto nasty_function = [](auto x) { return sin(x) + x; };
    auto result = solve_scalar_equation(nasty_function, 
                                        x0, tolerance, lower, upper);
    auto x = result.root;
    double error = std::abs(x);
    EXPECT_LT(error, tolerance);
}

TEST(ScalarEquationSolver, WorksWithTensorParameter)
{
    auto my_norm = [](auto A) {
        double tolerance = 1e-10;
        double lower = 1e-3;
        double upper = 20.0;
        double x0 = 10.0;
        auto sol = solve_scalar_equation([](auto x, auto P){ return x*x - squared_norm(P); }, x0, tolerance, lower, upper, A);
        return sol.root;
    };

    tensor< double, 3, 3 > A = {{{1.0, 2.0, 3.0}, 
                                 {4.0, 5.0, 6.0}, 
                                 {7.0, 8.0, 9.0}}};

    auto [normA, dnormA_dA] = my_norm(make_dual(A));
    double exact_value = norm(A);
    EXPECT_LT(std::abs(normA - exact_value)/exact_value, 1e-12);

    tensor<double, 3, 3> exact_derivative = A/norm(A);
    EXPECT_LT(norm(dnormA_dA - exact_derivative)/norm(exact_derivative), 1e-12);
}

TEST(ScalarEquationSolver, CanTakeDirectionalDerivative)
{
    auto my_norm = [](auto A) {
        double tolerance = 1e-10;
        double lower = 1e-3;
        double upper = 20.0;
        double x0 = 10.0;
        auto sol = solve_scalar_equation([](auto x, auto P){ return x*x - squared_norm(P); }, x0, tolerance, lower, upper, A);
        return sol.root;
    };

    tensor< double, 3, 3 > A = {{{1.0, 2.0, 3.0},
                                 {4.0, 5.0, 6.0},
                                 {7.0, 8.0, 9.0}}};

    tensor< double, 3, 3 > dA = {{{1.0, 2.0, 3.0},
                                  {4.0, 5.0, 6.0},
                                  {7.0, 8.0, 9.0}}};

    // this should give dx_dA : dA = A:A/norm(A) = norm(A)
    auto [normA, dnormA] = my_norm(make_dual(A, dA));

    double exact_derivative = norm(A);
    EXPECT_LT(std::abs(exact_derivative - dnormA)/exact_derivative, 1e-12);
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

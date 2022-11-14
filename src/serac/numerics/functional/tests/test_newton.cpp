// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include "serac/numerics/functional/tuple_tensor_dual_functions.hpp"

using namespace serac;

template <typename T>
T f(T x)
{
    return x*x - 2.0;
}

template <typename S, typename T>
auto g(S x, T p)
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
    auto result = solve_scalar_equation([](auto x){ return f(x);}, x0, tolerance, lower, upper);
    auto x = result.root;
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
    auto result = solve_scalar_equation([](auto x, auto a){ return g(x, a);}, x0, tolerance, lower, upper, p);
    auto x = result.root;
    double error = std::abs((x - std::sqrt(p))/std::sqrt(p));
    EXPECT_LT(error, tolerance);
}

TEST(ScalarEquationSolver, DerivativeOfPrimal)
{
    auto my_sqrt = [](dual<double> p) {
        double x0 = get_value(p);
        double tolerance = 1e-6;
        double lower = 0;
        double upper = (get_value(p) > 1.0) ? get_value(p) : 1.0;
        auto result = solve_scalar_equation([](auto x, auto a){ return g(x, a); }, x0, tolerance, lower, upper, p);
        return result.root;
    };
    double p = 2.0;
    auto [sqrt_p, dsqrt_p] = my_sqrt(make_dual(p));
    EXPECT_NEAR(dsqrt_p, 0.5/std::sqrt(p), 1e-12);
}

TEST(ScalarEquationSolver, AbortsIfRootNotBracketedByCaller)
{
    double x0 = 5.0;
    double tolerance = 1e-8;
    double lower = 2.0;
    double upper = 10.0;
    EXPECT_DEATH_IF_SUPPORTED({
            [[maybe_unused]] auto result = 
                solve_scalar_equation([](auto x){ return f(x); }, x0, tolerance, lower, upper);
        }, "solve_scalar_equation: root not bracketed by input bounds.");
}

TEST(ScalarEquationSolver, ReturnsImmediatelyIfUpperBoundIsARoot)
{
    double p = 4.0;
    double upper = 2.0;
    
    double x0 = 1.0;
    double tolerance = 1e-8;
    double lower = 0.0;

    auto result = solve_scalar_equation([](auto x, auto a){ return g(x, a);}, x0, tolerance, lower, upper, p);

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

    auto result = solve_scalar_equation([](auto x, auto a){ return g(x, a);}, x0, tolerance, lower, upper, p);

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
    auto result = solve_scalar_equation([](auto x) { return function_that_kills_newton(x); }, 
                                        x0, tolerance, lower, upper);
    auto x = result.root;
    double error = std::abs(x);
    EXPECT_LT(error, tolerance);
}

// TEST(ScalarEquationSolver, DerivativeWithTensorParameter)
// {
//     auto my_norm = [](auto A) {
//         auto p = squared_norm(A);
//         double tolerance = 1e-10;
//         double lower = 1e-3;
//         double upper = 20.0;
//         double x0 = get_value(p);
//         return solve_scalar_equation([](auto x, auto a){ return g(x, a); }, x0, tolerance, lower, upper, p);
//     };

//     tensor< double, 3, 3 > A = {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}};

//     auto [my_norm_p, dmy_norm_p] = my_norm(make_dual(A));
//     auto [norm_p, dnorm_p] = norm(make_dual(A));
//     std::cout << "my_norm_p  = " << my_norm_p << std::endl;
//     std::cout << "norm_p     = " << norm_p << std::endl;
//     std::cout << "dmy_norm_p = " << dmy_norm_p << std::endl;
//     std::cout << "dnorm_p    = " << dnorm_p << std::endl;
// }

TEST(ScalarEquationSolver, DerivativeWithTensorParameter2)
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

    // auto [my_norm_p, dmy_norm_p] = my_norm(make_dual(A));
    auto my_norm_p = my_norm(make_dual(A));
    auto [norm_p, dnorm_p] = norm(make_dual(A));
    std::cout << "my_norm_p  = " << my_norm_p << std::endl;
    std::cout << "norm_p     = " << norm_p << std::endl;
    //std::cout << "dmy_norm_p = " << dmy_norm_p << std::endl;
    std::cout << "dnorm_p    = " << dnorm_p << std::endl;
}

TEST(ScalarEquationSolver, dummy)
{
    auto f = [](auto c, auto v, auto w) { return dot(v, w)*c; };

    double x = 2.0;
    tensor<double, 2> v{{1.0, 2.0}};
    tensor<double, 2> w{{3.0, 4.0}};
    auto v_dual = make_dual(v);
    auto w_dual = make_dual(w);
    auto r = f(x, v_dual, w_dual);
    std::cout << "r val = " << get_value(r) << std::endl;
    std::cout << "r grad = " << get_gradient(r) << std::endl;
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

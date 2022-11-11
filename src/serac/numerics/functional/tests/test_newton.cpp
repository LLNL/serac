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

// TEST(ScalarEquationSolver, Converges)
// {
//     double x0 = 2.0;
//     double tolerance = 1e-8;
//     double lower = 1e-3;
//     double upper = 2.5;
//     double x = solve_scalar_equation([](auto x){ return f(x);}, x0, tolerance, lower, upper);
//     double error = std::abs((x - std::sqrt(2.0))/std::sqrt(2.0));
//     EXPECT_LT(error, tolerance);
// }

// TEST(ScalarEquationSolver, WorksWithParameter)
// {
//     double x0 = 2.0;
//     double tolerance = 1e-8;
//     double lower = 1e-3;
//     double upper = 2.5;
//     double p = 2.0;
//     double x = solve_scalar_equation([](auto x, auto a){ return g(x, a);}, x0, tolerance, lower, upper, p);
//     double error = std::abs((x - std::sqrt(p))/std::sqrt(p));
//     EXPECT_LT(error, tolerance);
// }

TEST(ScalarEquationSolver, DerivativeOfPrimal)
{
    auto my_sqrt = [](dual<double> p) {
        double x0 = get_value(p);
        double tolerance = 1e-10;
        double lower = 1e-3;
        double upper = 10.0;
        return solve_scalar_equation([](auto x, auto a){ return g(x, a); }, x0, tolerance, lower, upper, p);
    };
    double p = 2.0;
    auto [sqrt_p, dsqrt_p] = my_sqrt(make_dual(p));
    std::cout << "sqrt_p = " << sqrt_p << std::endl;
    std::cout << "dsqrt_p (AD)     = " << dsqrt_p << std::endl;
    std::cout << "dsqrt_p (exact)) = " << 0.5 / (sqrt_p) << std::endl;
}

TEST(ScalarEquationSolver, DerivativeWithTensorParameter)
{
    auto my_norm = [](auto A) {
        auto p = squared_norm(A);
        double tolerance = 1e-10;
        double lower = 1e-3;
        double upper = 20.0;
        double x0 = get_value(p);
        return solve_scalar_equation([](auto x, auto a){ return g(x, a); }, x0, tolerance, lower, upper, p);
    };

    tensor< double, 3, 3 > A = {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}};

    auto [my_norm_p, dmy_norm_p] = my_norm(make_dual(A));
    auto [norm_p, dnorm_p] = norm(make_dual(A));
    std::cout << "my_norm_p  = " << my_norm_p << std::endl;
    std::cout << "norm_p     = " << norm_p << std::endl;
    std::cout << "dmy_norm_p = " << dmy_norm_p << std::endl;
    std::cout << "dnorm_p    = " << dnorm_p << std::endl;
}

TEST(ScalarEquationSolver, DerivativeWithTensorParameter2)
{
    auto my_norm = [](auto A) {
        double tolerance = 1e-10;
        double lower = 1e-3;
        double upper = 20.0;
        double x0 = 10.0;
        return solve_scalar_equation([](auto x, auto P){ return x*x - squared_norm(P); }, x0, tolerance, lower, upper, A);
    };

    tensor< double, 3, 3 > A = {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}};

    auto [my_norm_p, dmy_norm_p] = my_norm(make_dual(A));
    auto [norm_p, dnorm_p] = norm(make_dual(A));
    std::cout << "my_norm_p  = " << my_norm_p << std::endl;
    std::cout << "norm_p     = " << norm_p << std::endl;
    std::cout << "dmy_norm_p = " << dmy_norm_p << std::endl;
    std::cout << "dnorm_p    = " << dnorm_p << std::endl;
}

    // def test_scalar_newton_jvp(self):
    //     def my_sqrt(p):
    //         tol = 1e-12
    //         x = newton.solve_scalar(f, p, tol, p)
    //         return x
        
    //     p = 2.0
    //     s = my_sqrt(p)
    //     print(f"{s:.14f}")
    //     dsqrt = jax.jacfwd(my_sqrt)
    //     print(f"{dsqrt(p):.14f}")
    //     error = np.abs(dsqrt(p) - 0.5/np.sqrt(p))
    //     self.assertLess(error, 1e-10)

// TEST(ScalarEquationSolver, AbortsIfRootNotBracketedByCaller)
// {
//     double x0 = 5.0;
//     double tolerance = 1e-8;
//     double lower = 2.0;
//     double upper = 10.0;
//     double x = solve_scalar_equation([](auto x){ return f(x);}, x0, tolerance, lower, upper);
//     std::cout << "x = " << x << std::endl;
// }

// TEST(ScalarEquationSolver, ConvergesWithGuessOutsideNewtonBasin)
// {
//     double x0 = 9.5;
//     double tolerance = 1e-8;
//     double lower = -10.0;
//     double upper = 10.0;
//     double x = solve_scalar_equation([](auto x) { return function_that_kills_newton(x); }, 
//                                      x0, tolerance, lower, upper);
//     double error = std::abs(x);
//     std::cout << "x = " << x << std::endl;
//     EXPECT_LT(error, tolerance);
// }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}

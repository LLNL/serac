#include "gtest/gtest.h"

#include "newton_solver_enzyme.hpp"
#include "enzyme.hpp"

double sqrt_res(double x, double a) { return x*x - a; }

namespace serac {

template <auto f>
double solve_scalar_enzyme_too(double x0, double param)
{
    auto fprime = [](double x, double p) {
        double dx = 1.0;
        return __enzyme_fwddiff<double>((void*)+f, enzyme_dup, x, dx, enzyme_const, p);
    };

    const int KMAX_ITERS = 25;
    double tolerance = 1e-10;

    double x = x0;
    [[maybe_unused]] bool converged = false;
    for (int i = 0; i < KMAX_ITERS; i++) {
        double r = f(x, param);
        if (std::abs(r) < tolerance) {
            converged = true;
            break;
        }
        double J = fprime(x, param);
        x -= r/J;
    }
    return x;
}

template <auto f>
double solve_scalar_enzyme_fwddiff(double x0, double dx0, double p, double dp)
{
    double x = solve_scalar_enzyme_too<f>(x0, p);
    double dfdx = __enzyme_fwddiff<double>((void*)f, enzyme_dup, x, 1.0, enzyme_const, p);
    double dfdp = __enzyme_fwddiff<double>((void*)f, enzyme_const, x, enzyme_dup, p, dp);
    std::cout << "Custom diff is being called" << std::endl;
    return -dfdp/dfdx;
}

// how to instantiate one of these in a way that enzyme sees it?
template <auto f>
void* __enzyme_register_derivative_solve_scalar_enzyme[] = { (void*) solve_scalar_enzyme_too<sqrt_res>, (void*) solve_scalar_enzyme_fwddiff<sqrt_res> };



} // namespace serac

TEST(EnzymeNewton, Solves)
{
    double sqrt4 = serac::solve_scalar_enzyme_too<sqrt_res>(1.0, 4.0);
    EXPECT_NEAR(sqrt4, 2.0, 1e-9);
}

TEST(EnzymeNewton, DerivativeFunctionCorrectness)
{
    double y = serac::solve_scalar_enzyme_fwddiff<sqrt_res>(1.0, 0.0, 4.0, 1.0);
    double exact = 0.25;
    EXPECT_NEAR(y, exact, 1e-9);
}

TEST(EnzymeNewton, RegisteredDerivative)
{
    double x0 = 1.0;
    double y = __enzyme_fwddiff<double>((void*) serac::solve_scalar_enzyme_too<sqrt_res>, enzyme_const, x0, enzyme_dup, 4.0, 1.0);
    double exact = 0.25;
    EXPECT_NEAR(y, exact, 1e-9);
}

static double g(double x, double p) { return x*x - p; }

TEST(EnzymeNewton, StaticVersionSolves)
{
    double sqrt4 = serac::solve_scalar_enzyme(sqrt_res, 1.0, 4.0);
    EXPECT_NEAR(sqrt4, 2.0, 1e-9);
}

TEST(EnzymeNewton, WorksWithStatelessLambda)
{
    auto f = [](double x, double p) { return x*x - p; };
    double sqrt4 = serac::solve_scalar_enzyme(f, 1.0, 4.0);
    EXPECT_NEAR(sqrt4, 2.0, 1e-9);
}

#include "gtest/gtest.h"

#include "tensor.hpp"
#include "enzyme.hpp"

double sqrt_res(double x, double a) { return x*x - a; }

namespace serac {

template <auto f>
double solve_scalar_enzyme(double x0, double param)
{
    auto fprime = [=](double x, double p) {
        double dx = 1.0;
        return __enzyme_fwddiff<double>((void*)f, enzyme_dup, x, dx, enzyme_const, p);
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

} // namespace serac

TEST(EnzymeNewton, Solves)
{
    double (*f)(double, double) = &sqrt_res;
    // double y = f(2.0, 4.0);
    // std::cout << y << std::endl;
    double sqrt4 = serac::solve_scalar_enzyme<sqrt_res>(1.0, 4.0);
    EXPECT_NEAR(sqrt4, 2.0, 1e-9);
}
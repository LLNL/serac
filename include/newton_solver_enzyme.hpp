#include "enzyme.hpp"

#include <cmath>

namespace serac {

template <typename Function>
double solve_scalar_enzyme(Function f, double x0, double param)
{
    auto fprime = [=](double x, double p) {
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

} // namespace serac

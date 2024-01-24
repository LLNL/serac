#include "enzyme.hpp"
#include "tuple.hpp"

#include <cmath>

namespace serac {

template <typename Function, typename T>
double solve_scalar_enzyme(Function f, double x0, T params)
{
    auto fprime = [=](double x, T p) {
        double dx = 1.0;
        T dp{};
        return __enzyme_fwddiff<double>((void*)f, enzyme_dup, x, dx, enzyme_dup, p, dp);
    };

    const int KMAX_ITERS = 25;
    double tolerance = 1e-10;

    double x = x0;
    [[maybe_unused]] bool converged = false;
    for (int i = 0; i < KMAX_ITERS; i++) {
        double r = f(x, params);
        if (std::abs(r) < tolerance) {
            converged = true;
            break;
        }
        double J = fprime(x, params);
        x -= r/J;
    }
    return x;
}

template < typename T, typename ... arg_types >
auto wrapper(T obj, arg_types && ... args) {
    return obj(args ... );
}

template <typename Function, typename T>
double solve_scalar_enzyme_wrapped(Function f, double x0, T params)
{
    auto fprime = [=](double x, T p) {
        double dx = 1.0;
        T dp{};
        return __enzyme_fwddiff<double>((void*)wrapper<Function, double, T>, enzyme_const, (void*)&f, 
                                        enzyme_dup, &x, &dx, enzyme_dup, &p, &dp);
    };

    const int KMAX_ITERS = 25;
    double tolerance = 1e-10;

    double x = x0;
    [[maybe_unused]] bool converged = false;
    for (int i = 0; i < KMAX_ITERS; i++) {
        double r = f(x, params);
        if (std::abs(r) < tolerance) {
            converged = true;
            break;
        }
        double J = fprime(x, params);
        x -= r/J;
    }
    return x;
}

} // namespace serac

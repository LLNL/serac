#include "gtest/gtest.h"

#include <stdio.h>

#include "tuple.hpp"
#include "tensor.hpp"
#include "materials.hpp"

double const tolerance = 1e-9;

TEST(PackedMaterial, Elastic) {
    double E = 1.0;
    double nu = 0.3;
    double rho = 1.0;
    double sigma_y = 1.0;
    double n = 1.0;
    double H0 = E/20.0;
    double eps0 = sigma_y/(n*H0);
    serac::PowerLawHardening hardening{.sigma_y = sigma_y, .n = n, .eps0 = eps0};
    serac::J2Nonlinear<serac::PowerLawHardening> mat{.E = E, .nu = nu, .hardening = hardening, .density = rho};

    serac::tensor<double, 3, 3> H = {{{0.5, 0.0, 0.0},
                                      {0.0,-0.15, 0.0},
                                      {0.0, 0.0, -0.15}}};
    serac::J2Nonlinear<serac::PowerLawHardening>::State<double> Q{};
    auto sigma = mat(Q, H);
    EXPECT_NEAR(sigma[0][0], E*H[0][0], tolerance);
}

TEST(PackedMaterial, Plastic) {
    double E = 1.0;
    double nu = 0.3;
    double rho = 1.0;
    double sigma_y = 1.0;
    double n = 1.0;
    double H0 = E/20.0;
    double eps0 = sigma_y/(n*H0);
    serac::PowerLawHardening hardening{.sigma_y = sigma_y, .n = n, .eps0 = eps0};
    serac::J2Nonlinear<serac::PowerLawHardening> mat{.E = E, .nu = nu, .hardening = hardening, .density = rho};

    double e00 = 1.5;
    double ep = (E*e00 - sigma_y)/(E + H0);
    double eps11 = -(0.5 - nu)*ep - nu*e00;
    serac::tensor<double, 3, 3> H = {{{e00, 0.0, 0.0},
                                      {0.0, eps11, 0.0},
                                      {0.0, 0.0, eps11}}};
    serac::J2Nonlinear<serac::PowerLawHardening>::State<double> Q{};
    auto sigma = mat(Q, H);
    double exact = E/(E + H0)*(H0*e00 + sigma_y);
    EXPECT_NEAR(sigma[0][0], exact, tolerance);
}

TEST(PackedMaterial, Dual) {
    double E = 1.0;
    double nu = 0.3;
    double rho = 1.0;
    double sigma_y = 1.0;
    double n = 1.0;
    double H0 = E/20.0;
    double eps0 = sigma_y/(n*H0);
    serac::PowerLawHardening hardening{.sigma_y = sigma_y, .n = n, .eps0 = eps0};
    serac::J2Nonlinear<serac::PowerLawHardening> mat{.E = E, .nu = nu, .hardening = hardening, .density = rho};

    double e00 = 1.5;
    double ep = (E*e00 - sigma_y)/(E + H0);
    double eps11 = -(0.5 - nu)*ep - nu*e00;
    serac::tensor<double, 3, 3> H = {{{e00, 0.0, 0.0},
                                      {0.0, eps11, 0.0},
                                      {0.0, 0.0, eps11}}};
    serac::J2Nonlinear<serac::PowerLawHardening>::State<double> Q{};
    auto sigma = mat(Q, serac::make_dual(H));
    std::cout << "stress " << serac::get_value(sigma) << std::endl;;
    std::cout << "tangent " << serac::get_gradient(sigma) << std::endl;
}

namespace serac
{
template <typename T>
tensor<T, 9> flatten(tensor<T, 3, 3> A)
{
    tensor<T, 9> a;
    for (int i = 0, ij = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++, ij++) {
            a[ij] = A[i][j];
        }
    }
    return a;
}

} // namespace serac

template<typename T>
using FlatTensor = serac::tensor<T, 9>;

TEST(PackedMaterial, CopyDuals) {
    serac::tensor<double, 3, 3> A{{{0.08811183127305, 0.39309930391406, 0.4883504766262},
                                   {0.63572859141876, 0.70760631671599, 0.0638470704629},
                                   {0.66157764230736, 0.27521984698671, 0.28507832693025}}};
    auto B = serac::make_dual(A);
    std::cout << "B = " << B << std::endl;
    auto b = serac::flatten(B);
    std::cout << "b flat = " << b << std::endl;
    // auto C = serac::dot(serac::transpose(B), B);
    // std::cout << "C = " << C << std::endl;

    FlatTensor<double> c = serac::flatten(A);
}
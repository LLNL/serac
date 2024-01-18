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
    serac::J2Nonlinear<serac::PowerLawHardening>::State Q{};
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
    serac::J2Nonlinear<serac::PowerLawHardening>::State Q{};
    auto sigma = mat(Q, H);
    double exact = E/(E + H0)*(H0*e00 + sigma_y);
    EXPECT_NEAR(sigma[0][0], exact, tolerance);
}
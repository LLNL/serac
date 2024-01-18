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
    double n = 2.0;
    double eps0 = sigma_y/(n*E/20.0);
    serac::PowerLawHardening hardening{.sigma_y = sigma_y, .n = n, .eps0 = eps0};
    serac::J2Nonlinear<serac::PowerLawHardening> mat{.E = E, .nu = nu, .hardening = hardening, .density = rho};

    serac::tensor<double, 3, 3> H = {{{0.5, 0.0, 0.0},
                                      {0.0,-0.15, 0.0},
                                      {0.0, 0.0, -0.15}}};
    serac::J2Nonlinear<serac::PowerLawHardening>::State Q{};
    auto sigma = mat(Q, H);
    EXPECT_NEAR(sigma[0][0], E*H[0][0], tolerance);
}
// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file nonlinear_J2_material.cpp
 */

#include "serac/physics/materials/solid_material.hpp"

#include <gtest/gtest.h>

#include "serac/numerics/functional/tensor.hpp"
#include "serac/physics/materials/material_verification_tools.hpp"

namespace serac {

TEST(NonlinearJ2Material, PowerLawHardeningWorksWithDuals)
{
  double sigma_y = 1.0;
  solid_mechanics::PowerLawHardening hardening_law{.sigma_y = sigma_y, .n=2.0, .eps0=0.01};
  double eqps = 0.1;
  auto flow_stress = hardening_law(make_dual(eqps));
  EXPECT_GT(flow_stress.value, sigma_y);
  EXPECT_GT(flow_stress.gradient, 0.0);
};

TEST(NonlinearJ2Material, SatisfiesConsistency)
{
  tensor<double, 3, 3> du_dx{{{0.7551559 , 0.3129729 , 0.12388372},
                              {0.548188  , 0.8851279 , 0.30576992},
                              {0.82008433, 0.95633745, 0.3566252 }}};
  solid_mechanics::PowerLawHardening hardening_law{.sigma_y = 0.1, .n=2.0, .eps0=0.01};
  solid_mechanics::J2Nonlinear<solid_mechanics::PowerLawHardening> material{.E = 1.0, .nu=0.25, .hardening=hardening_law, .density=1.0};
  auto internal_state = solid_mechanics::J2Nonlinear<solid_mechanics::PowerLawHardening>::State{};
  tensor<double, 3, 3> stress = material(internal_state, du_dx);
  double mises = std::sqrt(1.5)*norm(dev(stress));
  double flow_stress = hardening_law(internal_state.accumulated_plastic_strain);
  EXPECT_NEAR(mises, flow_stress, 1e-9*mises);

  double twoG = material.E/(1 + material.nu);
  tensor<double, 3, 3> s = twoG*dev(sym(du_dx) - internal_state.plastic_strain);
  EXPECT_LT(norm(s - dev(stress))/norm(s), 1e-9);
};

TEST(NonlinearJ2Material, Uniaxial)
{
  double E = 1.0;
  double nu = 0.25;
  double sigma_y = 0.01;
  double Hi = E/100.0;
  double eps0 = sigma_y/Hi;
  double n = 1;
  solid_mechanics::PowerLawHardening hardening{.sigma_y=sigma_y, .n=n, .eps0=eps0};
  solid_mechanics::J2Nonlinear<decltype(hardening)> material{.E=E, .nu=nu, .hardening=hardening, .density=1.0};

  auto internal_state = solid_mechanics::J2Nonlinear<decltype(hardening)>::State{};
  auto strain = [=](double t) { return sigma_y/E*t; };
  auto response_history = uniaxial_stress_test(2.0, 3, material, internal_state, strain);

  auto stress_exact = [=](double strain) { 
    return strain < sigma_y/E ? E*strain : E/(E + Hi)*(sigma_y + Hi*strain);
  };
  auto plastic_strain_exact = [=](double strain) {
    return strain < sigma_y/E ? E*strain : (E*strain - sigma_y)/(E + Hi);
  };

  for (auto r : response_history) {
    double e = get<1>(r)[0][0]; // strain
    double s = get<2>(r)[0][0]; // stress
    double pe = get<3>(r).plastic_strain[0][0]; // plastic strain
    ASSERT_LE(std::abs(s - stress_exact(e)), 1e-10*std::abs(stress_exact(e)));
    ASSERT_LE(std::abs(pe - plastic_strain_exact(e)), 1e-10*std::abs(plastic_strain_exact(e)));
  }
};


} // namespace serac


int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();

  return result;
}
// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file nonlinear_J2_material.cpp
 */

#include "serac/physics/materials/solid_material.hpp"

#include <iostream>
#include <fstream>

#include <gtest/gtest.h>

#include "serac/numerics/functional/tensor.hpp"
#include "serac/physics/materials/material_verification_tools.hpp"

namespace serac {

TEST(NonlinearJ2Material, PowerLawHardening)
{
  solid_mechanics::PowerLawHardening hardening_law{.sigma_y = 1.0, .n=2.0, .eps0=0.01};
  std::ofstream file;
  file.open("power_law_hardening.csv", std::ios::in | std::ios::trunc);

  double eqps = 0.0;
  for (size_t i = 0; i < 50; i++) {
      auto stress = hardening_law(make_dual(eqps));
      file << eqps << " " << stress.value << " " << stress.gradient << std::endl;
      eqps += 0.01;
  }
};

TEST(NonlinearJ2Material, Stress)
{
  tensor<double, 3, 3> du_dx{{{0.2, 0.0, 0.0},
                              {0.0, -0.05, 0.0},
                              {0.0, 0.0, -0.05}}};

  solid_mechanics::PowerLawHardening hardening_law{.sigma_y = 0.1, .n=2.0, .eps0=0.01};
  solid_mechanics::J2Nonlinear<solid_mechanics::PowerLawHardening> material{.E = 1.0, .nu=0.25, .hardening=hardening_law, .density=1.0};
  auto internal_state = solid_mechanics::J2Nonlinear<solid_mechanics::PowerLawHardening>::State{};
  auto stress = material(internal_state, make_dual(du_dx));
  EXPECT_GT(internal_state.accumulated_plastic_strain, 0);
  EXPECT_GE(norm(stress), 1e-4);
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
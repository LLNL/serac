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

namespace serac {

TEST(NonlinearJ2Material, Hardening)
{
  solid_mechanics::Hardening hardening_law{.sigma_y = 1.0, .n=2.0, .eps0=0.01};
  std::ofstream file;
  file.open("stress.csv", std::ios::in | std::ios::trunc);

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

  solid_mechanics::Hardening hardening_law{.sigma_y = 0.1, .n=2.0, .eps0=0.01};
  solid_mechanics::J2Nonlinear material{.E = 1.0, .nu=0.25, .hardening=hardening_law, .density=1.0};
  auto internal_state = solid_mechanics::J2Nonlinear::State{};
  auto stress = material(internal_state, make_dual(du_dx));
  std::cout << stress << std::endl;
  std::cout << internal_state.plastic_strain << std::endl;
  EXPECT_GE(norm(stress), 1e-4);
};

} // namespace serac


int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();

  return result;
}
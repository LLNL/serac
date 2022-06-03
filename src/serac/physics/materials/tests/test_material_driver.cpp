// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file test_material_point_driver.cpp
 *
 * @brief unit tests for the material point test utility
 */

#include <gtest/gtest.h>
#include <fstream>

#include "serac/physics/materials/material_driver.hpp"
#include "serac/physics/materials/solid_functional_material.hpp"

namespace serac {

TEST(MaterialDriver, testUniaxialTensionOnLinearMaterial)
{
  double density = 1.0;
  double E = 1.0;
  double nu = 0.25;
  double G = 0.5*E/(1.0 + nu);
  double K = E/3.0/(1.0 - 2.0*nu);
  solid_util::LinearIsotropicSolid<3> material(density, G, K);
  solid_util::MaterialDriver material_driver(material);
  double max_strain = 1.0;
  unsigned int steps = 10;
  auto response_history = material_driver.runUniaxial(max_strain, steps);
  
  for (unsigned int i = 0; i < response_history.size(); i++) {
    double computed_stress = get<1>(response_history[i]);
    double strain = get<0>(response_history[i]);
    double expected_stress = E*strain;
    //std::cout << strain << " " << computed_stress << std::endl;
    EXPECT_NEAR(computed_stress, expected_stress, 1e-8);
  }
}

TEST(MaterialDriver, testUniaxialTensionOnNonLinearMaterial)
{
  double density = 1.0;
  double E = 1.0;
  double nu = 0.25;
  double G = 0.5*E/(1.0 + nu);
  double K = E/3.0/(1.0 - 2.0*nu);
  solid_util::NeoHookeanSolid<3> material(density, G, K);
  solid_util::MaterialDriver material_driver(material);
  double max_strain = 1.0;
  unsigned int steps = 10;
  auto response_history = material_driver.runUniaxial(max_strain, steps);
  
  for (unsigned int i = 0; i < response_history.size(); i++) {
    double computed_stress = get<1>(response_history[i]);
    double strain = get<0>(response_history[i]);
    //std::cout << strain << " " << computed_stress << std::endl;
  }
}

} // namespace serac

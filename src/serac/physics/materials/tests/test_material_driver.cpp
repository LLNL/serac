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
#include "serac/physics/materials/parameterized_solid_functional_material.hpp"

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
  double max_time = 1.0;
  unsigned int steps = 10;
  const double strain_rate = 1.0;
  std::function<double(double)> constant_eng_strain_rate = [strain_rate](double t){ return strain_rate*t; };
  auto response_history = material_driver.runUniaxial(max_time, steps, constant_eng_strain_rate);

  for (const auto& r : response_history) {
    double computed_stress = get<1>(r);
    double strain = get<0>(r);
    double expected_stress = E*strain;
    EXPECT_NEAR(computed_stress, expected_stress, 1e-10);
    // std::cout << strain << " " << computed_stress << std::endl;
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
  double max_time = 1.0;
  unsigned int steps = 10;
  double strain_rate = 1.0;
  std::function<double(double)> constant_true_strain_rate = [strain_rate](double t){ return std::expm1(strain_rate*t); };
  auto response_history = material_driver.runUniaxial(max_time, steps, constant_true_strain_rate);

  for (const auto& r : response_history) {
    double computed_stress = get<1>(r);
    double strain = get<0>(r);
    std::cout << strain << " " << computed_stress << std::endl;
  }

}

TEST(MaterialDriver, testUniaxialTensionOnParameterizedMaterial)
{
  double density = 1.0;
  double E = 1.0;
  double nu = 0.25;
  double G = 0.5*E/(1.0 + nu);
  double K = E/3.0/(1.0 - 2.0*nu);
  solid_util::ParameterizedLinearIsotropicSolid<3> material(density, G, K);
  auto material_with_params = [&material](auto x, auto u, auto dudx, auto state)
  {
    return material(x, u, dudx, state, 0.0, 0.0);
  };
  solid_util::MaterialDriver material_driver(material_with_params);
  double max_time = 1.0;
  unsigned int steps = 10;
  const double strain_rate = 1.0;
  std::function<double(double)> constant_eng_strain_rate = [strain_rate](double t){ return strain_rate*t; };
  auto response_history = material_driver.runUniaxial(max_time, steps, constant_eng_strain_rate);

  for (const auto& r : response_history) {
    double computed_stress = get<1>(r);
    double strain = get<0>(r);
    double expected_stress = E*strain;
    EXPECT_NEAR(computed_stress, expected_stress, 1e-10);
    // std::cout << strain << " " << computed_stress << std::endl;
  }
}

TEST(MaterialDriver, testUniaxialTensionOnMaterialWithState)
{
  double density = 1.0;
  double E = 1.0;
  double nu = 0.25;
  double G = 0.5*E/(1.0 + nu);
  double K = E/3.0/(1.0 - 2.0*nu);
  solid_util::J2 material(density, G, K);
  solid_util::MaterialDriver material_driver(material);
  double max_time = 1.0;
  unsigned int steps = 10;
  const double strain_rate = 1.0;
  std::function<double(double)> constant_eng_strain_rate = [strain_rate](double t){ return strain_rate*t; };
  solid_util::MaterialState<J2> state{0.0, DenseIdentity<3>()};
  auto response_history = material_driver.runUniaxial(max_time, steps, constant_eng_strain_rate, state);

  for (const auto& r : response_history) {
    double computed_stress = get<1>(r);
    double strain = get<0>(r);
    double expected_stress = E*strain;
    EXPECT_NEAR(computed_stress, expected_stress, 1e-10);
    // std::cout << strain << " " << computed_stress << std::endl;
  }
}



} // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();

  return result;
}

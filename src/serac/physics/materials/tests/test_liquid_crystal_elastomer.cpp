// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file test_liquid_crystal_elastomer.cpp
 *
 * @brief unit tests for the Brighenti liquid crystal elastomer model
 */

#include <gtest/gtest.h>
#include <fstream>

#include "serac/physics/materials/material_verification_tools.hpp"
#include "serac/physics/materials/liquid_crystal_elastomer.hpp"
#include "serac/physics/materials/solid_functional_material.hpp" // for neohookean comparison

namespace serac {

TEST(TestLiquidCrystalMaterial, agreesWithNeoHookeanInHighTemperatureLimit)
{
  double density = 1.0;
  double E = 1.0;
  double nu = 0.49;
  double order_constant = 1.0;
  double order_parameter = 0.0;
  double transition_temperature = 1.0;
  tensor<double, 3> normal{{1.0, 0.0, 0.0}};
  double Nb2 = 1.0;
  double shear_modulus = 0.5*E/(1.0 + nu);
  double bulk_modulus = E / 3.0 / (1.0 - 2.0*nu);  

  BrighentiMechanical material(density, shear_modulus, bulk_modulus, order_constant,
                               order_parameter, transition_temperature, normal, Nb2);

  const tensor<double, 3> x{};
  const tensor<double, 3> u{};
  tensor<double, 3, 3> H{{{0.5625, 0.0, 0.0},
                          {0.0, -0.2, 0.0},
                          {0.0, 0.0, -0.2}}};
  double theta = 300.0;

  auto F_old = DenseIdentity<3>();
  double theta_old = 300.0;
  tensor<double, 3, 3> mu_old = BrighentiMechanical::calculateInitialDistributionTensor(normal, order_parameter, Nb2);
  BrighentiMechanical::State state{F_old, mu_old, theta_old};

  auto response = material(x, u, H, state, theta);
  std::cout << response.stress << std::endl;

  solid_util::NeoHookeanSolid<3> nh_material(density, shear_modulus, bulk_modulus);
  solid_util::NeoHookeanSolid<3>::State nh_state{};
  auto nh_response = nh_material(x, u, H, nh_state);
  std::cout << nh_response.stress << std::endl;
}
#if 0
TEST(TestLiquidCrystalMaterial, generatesStressHistory)
{
  double density = 1.0;
  double E = 1.0;
  double nu = 0.25;
  double shear_modulus = 0.5*E/(1.0 + nu);
  double bulk_modulus = E / 3.0 / (1.0 - 2.0*nu);
  double order_constant = 1.0;
  double order_parameter = 1.0;
  double transition_temperature = 1.0;
  tensor<double, 3> normal{{0.0, 1.0, 0.0}};
  double Nb2 = 1.0;
  BrighentiMechanical material(density, shear_modulus, bulk_modulus, order_constant, order_parameter, transition_temperature, normal, Nb2);
  double temperature = 2.0;
  auto initial_distribution = decltype(material)::calculateInitialDistributionTensor(normal, order_parameter, Nb2);
  decltype(material)::State initial_state{DenseIdentity<3>(), initial_distribution, temperature};
  double max_time = 20.0;
  unsigned int steps = 10;
  double strain_rate = 1e-2;
  std::function<double(double)> constant_strain_rate = [strain_rate](double t){ return strain_rate*t; };
  std::function<double(double)> constant_temperature = [temperature](double){ return temperature; };
  auto response_history = uniaxial_stress_test(max_time, steps, material, initial_state, constant_strain_rate, constant_temperature);

  for (auto const& [t, strain, stress, state] : response_history) {
    std::cout<< strain[0][0] << " " << stress << std::endl;
  }

  solid_util::NeoHookeanSolid<3> nh_material(density, shear_modulus, bulk_modulus);
  solid_util::NeoHookeanSolid<3>::State nh_state{};
  auto nh_response_history = uniaxial_stress_test(max_time, steps, nh_material, nh_state, constant_strain_rate);
  for (auto const& [t, strain, stress, state] : nh_response_history) {
    std::cout<< strain[0][0] << " " << stress << std::endl;
  }
}
#endif
} // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();

  return result;
}

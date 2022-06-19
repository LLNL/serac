// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file test_material_verification_tools.cpp
 *
 * @brief unit tests for the material point testing utilities
 */

#include <gtest/gtest.h>

#include "serac/physics/materials/material_verification_tools.hpp"
#include "serac/physics/materials/solid_functional_material.hpp"
#include "serac/physics/materials/parameterized_solid_functional_material.hpp"

namespace serac {

static constexpr double density = 1.0;
static constexpr double E = 1.0;
static constexpr double nu = 0.25;
static constexpr double G = 0.5*E/(1.0 + nu);
static constexpr double K = E/3.0/(1.0 - 2.0*nu);


TEST(MaterialVerificationTools, testUniaxialTensionOnLinearMaterial)
{
  solid_util::LinearIsotropicSolid<3> material(density, G, K);
  decltype(material)::State initial_state{};
  double max_time = 1.0;
  unsigned int steps = 10;
  const double strain_rate = 1.0;
  std::function<double(double)> prescribed_strain = [strain_rate](double t){ return strain_rate*t; };
  auto response_history = uniaxial_stress_test(max_time, steps, material, initial_state, prescribed_strain);

  for (const auto& [time, strain, stress, state] : response_history) {
    EXPECT_NEAR(stress[0][0], E * strain[0][0], 1e-10);
  }
}

TEST(MaterialVerificationTools, testUniaxialTensionOnNonLinearMaterial)
{
  solid_util::NeoHookeanSolid<3> material(density, G, K);
  decltype(material)::State initial_state{};
  double max_time = 1.0;
  unsigned int steps = 10;
  double strain_rate = 1.0;
  // constant true strain rate extension
  std::function<double(double)> constant_true_strain_rate = [strain_rate](double t){ return std::expm1(strain_rate*t); };
  auto response_history = uniaxial_stress_test(max_time, steps, material, initial_state, constant_true_strain_rate);

  for (const auto& [time, strain, stress, state] : response_history) {
    EXPECT_GE(stress[0][0], E*strain[0][0]);
    // check for uniaxial state
    EXPECT_LT(stress[1][1], 1e-10);
    EXPECT_LT(stress[2][2], 1e-10);
  }
}

TEST(MaterialVerificationTools, UniaxialTensionWithTimeIndependentParameters)
{
  solid_util::ParameterizedLinearIsotropicSolid<3> material(density, G, K);
  auto material_with_params = [&material](auto x, auto u, auto dudx, auto & state)
  {
    return material(x, u, dudx, state, 0.0, 0.0);
  };
  decltype(material)::State initial_state{};
  double max_time = 1.0;
  unsigned int steps = 10;
  const double strain_rate = 1.0;
  std::function<double(double)> constant_eng_strain_rate = [strain_rate](double t){ return strain_rate*t; };
  auto response_history = uniaxial_stress_test(max_time, steps, material_with_params, initial_state, constant_eng_strain_rate);

  for (const auto& [time, strain, stress, state] : response_history) {
    EXPECT_NEAR(stress[0][0], E * strain[0][0], 1e-10);
  }
}

TEST(MaterialVerificationTools, UniaxialTensionWithTimeDependentParameters)
{
  // In this test, the elastic module are modified as known functions of time.
  // This is weird from a physics standpoint, but it does let us verify
  // that the uniaxial_stress_test function handles time-dependent parameters
  // in a mathematically correct way.
  solid_util::ParameterizedLinearIsotropicSolid<3> material(density, G, K);
  decltype(material)::State initial_state{};
  double max_time = 1.0;
  unsigned int steps = 10;
  const double strain_rate = 1.0;
  std::function<double(double)> constant_eng_strain_rate = [strain_rate](double t){ return strain_rate*t; };
  auto DeltaG = [](double t){ return 1.0 + t; };
  auto DeltaK = [](double t){ return 1.0 + 3.0 * t; };
  auto response_history = uniaxial_stress_test(max_time, steps, material, initial_state, constant_eng_strain_rate, DeltaK, DeltaG);

  for (const auto& [time, strain, stress, state] : response_history) {
    double Gt = G + DeltaG(time);
    double Kt = K + DeltaK(time);
    double Et = 9.0*Kt*Gt/(3.0*Kt + Gt);
    EXPECT_NEAR(stress[0][0], Et * strain[0][0], 1e-10);
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

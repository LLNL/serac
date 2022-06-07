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

static constexpr double density = 1.0;
static constexpr double E = 1.0;
static constexpr double nu = 0.25;
static constexpr double G = 0.5*E/(1.0 + nu);
static constexpr double K = E/3.0/(1.0 - 2.0*nu);

template < typename MaterialType, typename StateType >
auto uniaxial_stress_test(
  std::function<double(double)> epsilon_xx,
  double t_max,
  size_t num_steps,
  const MaterialType material,
  const StateType initial_state) {

  tensor<double, 3> unused{};

  double t = 0;

  auto state = initial_state;

  auto sigma_yy_and_zz = [&](auto x) {
    auto epsilon_yy = x[0];
    auto epsilon_zz = x[1];
    using T = decltype(epsilon_yy);
    tensor<T, 3, 3> du_dx{};
    du_dx[0][0] = epsilon_xx(t);
    du_dx[1][1] = epsilon_yy;
    du_dx[2][2] = epsilon_zz;
    auto copy = state;
    auto output = material(unused, unused, du_dx, copy);
    return tensor{{output.stress[1][1], output.stress[2][2]}};
  };

  std::vector< tuple< double, double > > stress_strain_history;
  stress_strain_history.reserve(num_steps);

  tensor<double, 3, 3> dudx{};
  const double dt = t_max / double(num_steps);
  for (size_t i = 0; i < num_steps; i++) {
    t += dt;

    auto initial_guess = tensor<double, 2>{};
    auto epsilon_yy_and_zz = find_root(sigma_yy_and_zz, initial_guess);
    dudx[0][0] = epsilon_xx(t);
    dudx[1][1] = epsilon_yy_and_zz[0];
    dudx[2][2] = epsilon_yy_and_zz[1];

    auto stress = material(unused, unused, dudx, state).stress;
    stress_strain_history.push_back(tuple{dudx[0][0], stress[0][0]});

  }

  return stress_strain_history;
}

TEST(MaterialDriver, testUniaxialTensionOnLinearMaterial)
{
  solid_util::LinearIsotropicSolid<3> material(density, G, K);
  solid_util::MaterialDriver material_driver(material);
  decltype(material)::State state{};
  double max_time = 1.0;
  unsigned int steps = 10;
  const double strain_rate = 1.0;
  std::function<double(double)> prescribed_strain = [strain_rate](double t){ return strain_rate*t; };
  auto response_history = uniaxial_stress_test(prescribed_strain, max_time, steps, material, state);

  for (const auto& [strain, stress] : response_history) {
    EXPECT_NEAR(stress, E * strain, 1e-10);
  }
}

TEST(MaterialDriver, testUniaxialTensionOnNonLinearMaterial)
{
  solid_util::NeoHookeanSolid<3> material(density, G, K);
  solid_util::MaterialDriver material_driver(material);
  decltype(material)::State state{};
  double max_time = 1.0;
  unsigned int steps = 10;
  double strain_rate = 1.0;
  std::function<double(double)> constant_true_strain_rate = [strain_rate](double t){ return std::expm1(strain_rate*t); };
  auto response_history = uniaxial_stress_test(constant_true_strain_rate, max_time, steps, material, state);

  // todo: figure out the actual expected stress for this test
  //for (const auto& [strain, stress] : response_history) {
  //  EXPECT_NEAR(stress, E * strain, 1e-10);
  //}
}

TEST(MaterialDriver, testUniaxialTensionOnParameterizedMaterial)
{
  using material_type = solid_util::ParameterizedLinearIsotropicSolid<3>;
  material_type material(density, G, K);
  auto material_with_params = [&material](auto x, auto u, auto dudx, auto & state)
  {
    return material(x, u, dudx, state, 0.0, 0.0);
  };
  solid_util::MaterialDriver material_driver(material_with_params);
  material_type::State state{};
  double max_time = 1.0;
  unsigned int steps = 10;
  const double strain_rate = 1.0;
  std::function<double(double)> constant_eng_strain_rate = [strain_rate](double t){ return strain_rate*t; };
  auto response_history = uniaxial_stress_test(constant_eng_strain_rate, max_time, steps, material_with_params, state);

  for (const auto& [strain, stress] : response_history) {
    EXPECT_NEAR(stress, E * strain, 1e-10);
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

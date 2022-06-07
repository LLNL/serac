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

template < typename function, int n >
auto find_root(function && f, tensor< double, n > x0) {

  static_assert(std::is_same_v< decltype(f(x0)), tensor< double, n > >, 
    "error: f(x) must have the same number of equations as unknowns");

  double epsilon = 1.0e-8;
  int max_iterations = 10;

  auto x = x0;
  std::cout << x;

  int k = 0;
  while (k++ < max_iterations) {
    auto output = f(make_dual(x));
    auto r = get_value(output);
    std::cout << ", " << norm(r) << std::endl;
    if (norm(r) < epsilon) break;
    auto J = get_gradient(output);
    x -= linear_solve(J, r);
    std::cout << x;
  }

  return x;

};

#if 0
template < typename MaterialType >
std::vector < std::pair< double, double > > uniaxial_stress_test(
  std::function<double(double)> epsilon_xx,
  double t_max,
  int num_steps,
  const MaterialType material,
  const solid_util::MaterialState<MaterialType> & initial_state) {

  tensor<double, 3> unused{};

  double t = 0;

  auto state = initial_state;
  auto material_with_state = [&state](auto du_dx){
    if constepxr () {

    } else {

    }
  };

  auto sigma_yy_and_zz = [=](auto x) {
    auto epsilon_yy = x[0];
    auto epsilon_zz = x[1];
    using T = decltype(epsilon_yy);
    tensor<T, 3, 3> du_dx{};
    du_dx[0][0] = epsilon_xx(t);
    du_dx[1][1] = epsilon_yy;
    du_dx[2][2] = epsilon_zz;
    auto output = material_with_state(du_dx);
    return tensor{{output.stress[1][1], output.stress[2][2]}};
  };

  // for output
  std::vector< tuple< double, double > > stress_strain_history;

  const double dt = maxTime / nsteps;
  for (unsigned int i = 0; i < nsteps; i++) {
    t += dt;

    auto epsilon_yy_and_zz = find_root(sigma_yy_and_zz, initial_guess);

    dudx = solveForUniaxialState(dudx, state, relative_tolerance, max_equilibrium_iterations);
    if constexpr (std::is_base_of_v< Empty, MaterialState<MaterialType> >) {
      auto response = material_(x, u, make_dual(dudx));
      auto stress = get_value(response.stress);
      //std::cout << "out of plane stress " << stress[1][1] << std::endl;
      stress_strain_history.emplace_back(tuple{dudx[0][0], stress[0][0]});
    } else {
      auto response = material_(x, u, make_dual(dudx), state);
      auto stress = get_value(response.stress);
      //std::cout << "out of plane stress " << stress[1][1] << std::endl;
      stress_strain_history.emplace_back(tuple{dudx[0][0], stress[0][0]});
    }
  }

  return stress_strain_history;
}
#endif

TEST(MaterialDriver, find_root)
{

  std::cout << find_root([](auto x){
    return 3 * x * x[0] - tensor< double, 3 >{1.0, 2.0, 3.0};
  }, tensor<double, 3>{1.0, 2.0, 3.0});

  std::cout << std::endl;
  std::cout << std::endl;

  double density = 1.0;
  double E = 1.0;
  double nu = 0.25;
  double G = 0.5*E/(1.0 + nu);
  double K = E/3.0/(1.0 - 2.0*nu);
  //solid_util::LinearIsotropicSolid<3> material(density, G, K);
  solid_util::NeoHookeanSolid<3> material(density, G, K);
  decltype(material)::State state{};

  tensor< double, 3 > unused{};
  double epsilon_xx = 0.5;
  tensor < double, 2 > initial_guess{-0.25, -0.25};

  find_root([=, &state](auto x) {
    auto epsilon_yy = x[0];
    auto epsilon_zz = x[1];
    using T = decltype(epsilon_yy);
    tensor<T, 3, 3> du_dx{};
    du_dx[0][0] = epsilon_xx;
    du_dx[1][1] = epsilon_yy;
    du_dx[2][2] = epsilon_zz;
    auto output = material(unused, unused, du_dx, state);
    return tensor{{output.stress[1][1], output.stress[2][2]}};
  }, initial_guess);

}

#if 1
TEST(MaterialDriver, testUniaxialTensionOnLinearMaterial)
{
  double density = 1.0;
  double E = 1.0;
  double nu = 0.25;
  double G = 0.5*E/(1.0 + nu);
  double K = E/3.0/(1.0 - 2.0*nu);
  solid_util::LinearIsotropicSolid<3> material(density, G, K);
  solid_util::MaterialDriver material_driver(material);
  decltype(material)::State state{};
  double max_time = 1.0;
  unsigned int steps = 10;
  const double strain_rate = 1.0;
  std::function<double(double)> constant_eng_strain_rate = [strain_rate](double t){ return strain_rate*t; };
  auto response_history = material_driver.runUniaxial(max_time, steps, constant_eng_strain_rate, state);

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
  decltype(material)::State state{};
  double max_time = 1.0;
  unsigned int steps = 10;
  double strain_rate = 1.0;
  std::function<double(double)> constant_true_strain_rate = [strain_rate](double t){ return std::expm1(strain_rate*t); };
  auto response_history = material_driver.runUniaxial(max_time, steps, constant_true_strain_rate, state);

  for (const auto& r : response_history) {
    double computed_stress = get<1>(r);
    double strain = get<0>(r);
    std::cout << strain << " " << computed_stress << std::endl;
  }

}

TEST(MaterialDriver, testUniaxialTensionOnParameterizedMaterial)
{
  using material_type = solid_util::ParameterizedLinearIsotropicSolid<3>;
  double density = 1.0;
  double E = 1.0;
  double nu = 0.25;
  double G = 0.5*E/(1.0 + nu);
  double K = E/3.0/(1.0 - 2.0*nu);
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
  auto response_history = material_driver.runUniaxial(max_time, steps, constant_eng_strain_rate, state);

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
  solid_util::J2::State state{0.0, DenseIdentity<3>()};
  auto response_history = material_driver.runUniaxial(max_time, steps, constant_eng_strain_rate, state);

  for (const auto& r : response_history) {
    double computed_stress = get<1>(r);
    double strain = get<0>(r);
    double expected_stress = E*strain;
    EXPECT_NEAR(computed_stress, expected_stress, 1e-10);
    // std::cout << strain << " " << computed_stress << std::endl;
  }
}
#endif

} // namespace serac

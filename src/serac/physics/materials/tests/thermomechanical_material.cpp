// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermomechanical_material.cpp
 *
 * @brief unit tests for a thermoelastic material model
 */

#include <iostream>

#include <gtest/gtest.h>

#include "serac/physics/materials/green_saint_venant_thermoelastic.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/tuple.hpp"
#include "serac/numerics/functional/tuple_arithmetic.hpp"


namespace serac {

static constexpr auto I = Identity<3>();

// An arbitrary rotation tensor for frame indifference tests
static const tensor<double, 3, 3> Q{{{0.852932108456964, 0.405577416697288, -0.328654495524269},
                                     {-0.51876824611259, 0.728725283562492, -0.44703351990878},
                                     {0.058192140263313, 0.551784758906877, 0.831953877717782}}};


TEST(ThermomechanicalMaterial, FreeEnergyIsZeroInReferenceState)
{
  GreenSaintVenantThermoelastic material{.rho = 1.0, .E = 100.0, .nu = 0.25, .C = 1.0, .alpha = 1.0e-3, .theta_ref = 300.0, .k = 1.0};
  tensor<double, 3, 3>  displacement_grad{};
  double                temperature = material.theta_ref;
  double                free_energy = material.calculateFreeEnergy(displacement_grad, temperature);
  EXPECT_NEAR(free_energy, 0.0, 1e-10);
}

TEST(ThermomechanicalMaterial, StressIsZeroInReferenceState)
{
  GreenSaintVenantThermoelastic material{.rho = 1.0, .E = 100.0, .nu = 0.25, .C = 1.0, .alpha = 1.0e-3, .theta_ref = 300.0, .k = 1.0};
  tensor<double, 3, 3>  displacement_grad{};
  double                temperature = material.theta_ref;
  tensor<double, 3>     temperature_grad{};
  GreenSaintVenantThermoelastic::State state{};
  auto                         displacement_grad_old = displacement_grad;
  double                       temperature_old       = temperature;
  double                       dt                    = 1.0;
  auto stress = material.calculateMechanicalConstitutiveOutputs(displacement_grad, temperature, temperature_grad, state,
                                                                displacement_grad_old, temperature_old, dt);
  EXPECT_NEAR(norm(stress), 0.0, 1e-10);
}

TEST(ThermomechanicalMaterial, FreeEnergyAndStressAgree)
{
  GreenSaintVenantThermoelastic material{.rho = 1.0, .E = 100.0, .nu = 0.25, .C = 1.0, .alpha = 1.0e-3, .theta_ref = 300.0, .k = 1.0};
  // clang-format off
  tensor<double, 3, 3>  displacement_grad{{{0.35490513, 0.60419905, 0.4275843},
                                           {0.23061597, 0.6735498,  0.43953657},
                                           {0.25099766, 0.27730572, 0.7678207}}};
  // clang-format on
  double                       temperature = 290.0;
  tensor<double, 3>            temperature_grad{0.87241435, 0.11105156, -0.27708054};
  GreenSaintVenantThermoelastic::State state{};
  tensor<double, 3, 3>         displacement_grad_old{};
  double                       temperature_old = temperature;
  double                       dt              = 1.0;
  auto energy_and_stress = material.calculateFreeEnergy(make_dual(displacement_grad), temperature);
  auto stress = material.calculateMechanicalConstitutiveOutputs(displacement_grad, temperature, temperature_grad, state,
                                                                displacement_grad_old, temperature_old, dt);
  auto error  = stress - get_gradient(energy_and_stress);
  EXPECT_NEAR(norm(error), 0.0, 1e-12);
}

TEST(ThermomechanicalMaterial, SatisfiesDissipationInequality)
{
  GreenSaintVenantThermoelastic material{.rho = 1.0, .E = 100.0, .nu = 0.25, .C = 1.0, .alpha = 1.0e-3, .theta_ref = 300.0, .k = 1.0};
  // clang-format off
  tensor<double, 3, 3>  displacement_grad{{{0.35490513, 0.60419905, 0.4275843},
                                           {0.23061597, 0.6735498,  0.43953657},
                                           {0.25099766, 0.27730572, 0.7678207}}};
  // clang-format on
  double                       temperature = 290.0;
  tensor<double, 3>            temperature_grad{0.87241435, 0.11105156, -0.27708054};
  double                       temperature_old = temperature;
  GreenSaintVenantThermoelastic::State state{};
  tensor<double, 3, 3>         displacement_grad_old{};
  double                       dt                 = 1.0;
  auto                         generalized_fluxes = material.calculateThermalConstitutiveOutputs(
      displacement_grad, temperature, temperature_grad, state, displacement_grad_old, temperature_old, dt);
  auto [heat_capacity, source, heat_flux] = generalized_fluxes;
  auto dissipation                        = -dot(heat_flux, temperature_grad) / temperature;
  EXPECT_TRUE(dissipation >= 0.0);
}

TEST(ThermomechanicalMaterial, IsFrameIndifferent)
{
  GreenSaintVenantThermoelastic material{.rho = 1.0, .E = 100.0, .nu = 0.25, .C = 1.0, .alpha = 1.0e-3, .theta_ref = 300.0, .k = 1.0};
  // clang-format off
  tensor<double, 3, 3>  displacement_grad{{{0.35490513, 0.60419905, 0.4275843},
                                           {0.23061597, 0.6735498,  0.43953657},
                                           {0.25099766, 0.27730572, 0.7678207}}};
  // clang-format on
  double                       temperature = 290.0;
  tensor<double, 3>            temperature_grad{0.87241435, 0.11105156, -0.27708054};
  double                       temperature_old = 300;
  GreenSaintVenantThermoelastic::State state{};
  tensor<double, 3, 3>         displacement_grad_old{};
  double                       dt = 1.0;

  auto generalized_fluxes = material.calculateConstitutiveOutputs(displacement_grad, temperature, temperature_grad,
                                                                  state, displacement_grad_old, temperature_old, dt);

  auto displacement_grad_transformed     = Q * (displacement_grad + I) - I;
  auto displacement_grad_old_transformed = Q * (displacement_grad_old + I) - I;
  auto generalized_fluxes_2 =
      material.calculateConstitutiveOutputs(displacement_grad_transformed, temperature, temperature_grad, state,
                                            displacement_grad_old_transformed, temperature_old, dt);

  auto [piola_stress, heat_capacity, internal_source, heat_flux]         = generalized_fluxes;
  auto [piola_stress_2, heat_capacity_2, internal_source_2, heat_flux_2] = generalized_fluxes_2;
  EXPECT_LT(norm(piola_stress - transpose(Q) * piola_stress_2), 1e-12);
  EXPECT_NEAR(heat_capacity, heat_capacity_2, 1e-12);
  EXPECT_NEAR(internal_source, internal_source_2, 1e-12);
  EXPECT_LT(norm(heat_flux - heat_flux_2), 1e-12);
}

TEST(ThermomechanicalMaterial, InternalSourceHasCorrectSign)
{
  GreenSaintVenantThermoelastic material{.rho = 1.0, .E = 100.0, .nu = 0.25, .C = 1.0, .alpha = 1.0e-3, .theta_ref = 300.0, .k = 1.0};
  // clang-format off
  tensor<double, 3, 3>  displacement_grad{{{0.35490513, 0.60419905, 0.4275843},
                                           {0.23061597, 0.6735498,  0.43953657},
                                           {0.25099766, 0.27730572, 0.7678207}}};
  // clang-format on
  double                       temperature_old = 290.0;
  tensor<double, 3>            temperature_grad{0.87241435, 0.11105156, -0.27708054};
  GreenSaintVenantThermoelastic::State state{};
  tensor<double, 3, 3>         displacement_grad_old{};
  double                       temperature         = temperature_old;
  double                       dt                  = 1.0;
  auto [heat_capacity, internal_source, heat_flux] = material.calculateThermalConstitutiveOutputs(
      displacement_grad, temperature, temperature_grad, state, displacement_grad_old, temperature_old, dt);
  // should have same sign as sgn(alpha*trEdot), here negative
  EXPECT_LT(internal_source, 0.0);
}

TEST(ThermomechanicalMaterial, StressHasCorrectSymmetry)
{
  GreenSaintVenantThermoelastic material{.rho = 1.0, .E = 100.0, .nu = 0.25, .C = 1.0, .alpha = 1.0e-3, .theta_ref = 300.0, .k = 1.0};
  // clang-format off
  tensor<double, 3, 3>  displacement_grad{{{0.35490513, 0.60419905, 0.4275843},
                                           {0.23061597, 0.6735498,  0.43953657},
                                           {0.25099766, 0.27730572, 0.7678207}}};
  // clang-format on
  double                       temperature_old = 290.0;
  tensor<double, 3>            temperature_grad{0.87241435, 0.11105156, -0.27708054};
  GreenSaintVenantThermoelastic::State state{};
  tensor<double, 3, 3>         displacement_grad_old{};
  double                       temperature  = temperature_old;
  double                       dt           = 1.0;
  auto                         piola_stress = material.calculateMechanicalConstitutiveOutputs(
      displacement_grad, temperature, temperature_grad, state, displacement_grad_old, temperature_old, dt);
  auto   deformation_grad = displacement_grad + I;
  auto   kirchhoff_stress = piola_stress * transpose(deformation_grad);
  double tol              = 1e-10;
  EXPECT_TRUE(is_symmetric(kirchhoff_stress, tol));
}

}  // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);

  axom::slic::SimpleLogger logger;
  
  int result = RUN_ALL_TESTS();

  return result;
}

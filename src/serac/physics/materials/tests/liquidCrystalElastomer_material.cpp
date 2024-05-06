// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
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

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>

#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/tuple.hpp"
#include "serac/physics/materials/liquid_crystal_elastomer.hpp"

namespace serac {

static constexpr auto I = Identity<3>();

// An arbitrary rotation tensor for frame indifference tests
static const tensor<double, 3, 3> Q{{{0.852932108456964, 0.405577416697288, -0.328654495524269},
                                     {-0.51876824611259, 0.728725283562492, -0.44703351990878},
                                     {0.058192140263313, 0.551784758906877, 0.831953877717782}}};

// TEST(LiquidCrystalElastomerMaterial, FreeEnergyIsZeroInReferenceState)
// {
//   // LiquidCrystalElastomerZhang material{.E = 100.0, .nu = 0.25, .C = 1.0, .alpha = 1.0e-3, .theta_ref = 300.0, .k = 1.0};

//   double density         = 1.0;
//   double young_modulus   = 9.34e5;
//   double possion_ratio   = 0.45;
//   double beta_param      = 5.75e5;
//   double max_order_param = 0.40;
//   LiquidCrystalElastomerZhang material(density, young_modulus, possion_ratio, max_order_param, beta_param);

//   tensor<double, 3, 3>  displacement_grad{};
//   double                temperature = material.theta_ref;
//   double                free_energy = material.calculateStrainEnergy(displacement_grad, temperature);
//   EXPECT_NEAR(free_energy, 0.0, 1e-10);
// }

// TEST(LiquidCrystalElastomerMaterial, StressIsZeroInReferenceState)
// {
//   // LiquidCrystalElastomerZhang material{.E = 100.0, .nu = 0.25, .C = 1.0, .alpha = 1.0e-3, .theta_ref = 300.0, .k = 1.0};
//   double density         = 1.0;
//   double young_modulus   = 9.34e5;
//   double possion_ratio   = 0.45;
//   double beta_param      = 5.75e5;
//   double max_order_param = 0.40;
//   LiquidCrystalElastomerZhang material(density, young_modulus, possion_ratio, max_order_param, beta_param);

//   tensor<double, 3, 3>  displacement_grad{};
//   double                temperature = material.theta_ref;
//   tensor<double, 3>     temperature_grad{};
//   LiquidCrystalElastomerZhang::State state{};
//   auto                         displacement_grad_old = displacement_grad;
//   double                       temperature_old       = temperature;
//   double                       dt                    = 1.0;
//   auto stress = material(displacement_grad, temperature, temperature_grad, state,
//                                                                 displacement_grad_old, temperature_old, dt);
//   EXPECT_NEAR(norm(stress), 0.0, 1e-10);
// }

TEST(LiquidCrystalElastomerMaterial, StrainEnergyAndStressAgree)
{
  // LiquidCrystalElastomerZhang material{.E = 100.0, .nu = 0.25, .C = 1.0, .alpha = 1.0e-3, .theta_ref = 300.0, .k = 1.0};
  double density         = 1.0;
  double shear_mod       = 3.113e5; //  young_modulus_ / 2.0 / (1.0 + poisson_ratio_);
  double ini_order_param = 0.40;
  double omega_param     = 0.12;
  double bulk_mod        = 100.0*shear_mod;
  LiquidCrystalElastomerZhang material(density, shear_mod, ini_order_param, omega_param, bulk_mod);

  // clang-format off
  tensor<double, 3, 3>  displacement_grad{{{0.35490513, 0.60419905, 0.4275843},
                                           {0.23061597, 0.6735498,  0.43953657},
                                           {0.25099766, 0.27730572, 0.7678207}}};
  // clang-format on
  LiquidCrystalElastomerZhang::State state{};

  auto order_param_tuple = serac::make_tuple(0.39, 0.0);
  auto gamma_param_tuple = serac::make_tuple(M_PI_2, 0.0);
  auto eta_param_tuple = serac::make_tuple(0.0, 0.0);

  // auto energy_and_stress = material.calculateStrainEnergy(make_dual(displacement_grad), temperature);
  auto energy_and_stress = material.calculateStrainEnergy(state, make_dual(displacement_grad), order_param_tuple, gamma_param_tuple, eta_param_tuple);
  auto CauchyStress = material(state, displacement_grad, order_param_tuple, gamma_param_tuple, eta_param_tuple);

  auto F    = displacement_grad + I;
  auto Finv = inv(F);
  auto J    = det(F);
  auto PKStress = J * CauchyStress * transpose(Finv);

  auto error  = PKStress - get_gradient(energy_and_stress);
  
  EXPECT_NEAR(norm(error), 0.0, 1e-10*norm(PKStress));
}

// TEST(LiquidCrystalElastomerMaterial, IsFrameIndifferent)
// {
//   // LiquidCrystalElastomerZhang material{.E = 100.0, .nu = 0.25, .C = 1.0, .alpha = 1.0e-3, .theta_ref = 300.0, .k = 1.0};
//   double density         = 1.0;
//   double young_modulus   = 9.34e5;
//   double possion_ratio   = 0.45;
//   double beta_param      = 5.75e5;
//   double max_order_param = 0.40;
//   LiquidCrystalElastomerZhang material(density, young_modulus, possion_ratio, max_order_param, beta_param);

//   // clang-format off
//   tensor<double, 3, 3>  displacement_grad{{{0.35490513, 0.60419905, 0.4275843},
//                                            {0.23061597, 0.6735498,  0.43953657},
//                                            {0.25099766, 0.27730572, 0.7678207}}};
//   // clang-format on
//   double                       temperature = 290.0;
//   tensor<double, 3>            temperature_grad{0.87241435, 0.11105156, -0.27708054};
//   double                       temperature_old = 300;
//   LiquidCrystalElastomerZhang::State state{};
//   tensor<double, 3, 3>         displacement_grad_old{};
//   double                       dt = 1.0;

//   auto generalized_fluxes = material(displacement_grad, temperature, temperature_grad,
//                                                                   state, displacement_grad_old, temperature_old, dt);

//   auto displacement_grad_transformed     = Q * (displacement_grad + I) - I;
//   auto displacement_grad_old_transformed = Q * (displacement_grad_old + I) - I;
//   auto generalized_fluxes_2 =
//       material(displacement_grad_transformed, temperature, temperature_grad, state,
//                                             displacement_grad_old_transformed, temperature_old, dt);

//   auto [piola_stress, heat_capacity, internal_source, heat_flux]         = generalized_fluxes;
//   auto [piola_stress_2, heat_capacity_2, internal_source_2, heat_flux_2] = generalized_fluxes_2;
//   EXPECT_LT(norm(piola_stress - transpose(Q) * piola_stress_2), 1e-12);
//   EXPECT_NEAR(heat_capacity, heat_capacity_2, 1e-12);
//   EXPECT_NEAR(internal_source, internal_source_2, 1e-12);
//   EXPECT_LT(norm(heat_flux - heat_flux_2), 1e-12);
// }
}  // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();

  return result;
}

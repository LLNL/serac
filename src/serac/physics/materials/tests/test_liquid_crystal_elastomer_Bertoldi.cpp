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
#include "serac/physics/materials/liquid_crystal_elastomer_material.hpp"
#include "serac/physics/materials/solid_material.hpp" // for neohookean comparison

namespace serac {

TEST(TestLiquidCrystalBertoldiMat, ConsistentStressDerivedFromStrainEnergy)
{
  double density = 1.0;
  double young_modulus = 1.0;
  double possion_ratio = 0.49;
  double beta_param = 1.0;
  double max_order_param = 0.5;

  tuple <double, int> order_param_tuple, gamma_param_tuple;
  order_param_tuple = make_tuple(0.1, 123);
  gamma_param_tuple = make_tuple(0.9, 456);

  LiqCrystElast_Bertoldi LCEMat_Bertoldi(density, young_modulus, possion_ratio, max_order_param, beta_param);

  // test conditions
  tensor<double, 3, 3> H{{{0.423,  0.11, -0.123},
                          {-0.19, 0.25, 0.0},
                          {0.0,  0.11 , 0.39}}};

  // liquid crystal elastomer model response
  LiqCrystElast_Bertoldi::State state{};
  auto stress = LCEMat_Bertoldi(state, H, order_param_tuple, gamma_param_tuple);

  // Transform Cauchy stress into Piola stress, which is what the AD computation returns
  auto I = Identity<3>();  
  auto F = H + I;
  auto J = det(F);
  auto P_stress = J*stress*inv(transpose(F));

  // Strain energy
  auto free_energy = LCEMat_Bertoldi.calculateStrainEnergy(state, make_dual(H), order_param_tuple, gamma_param_tuple);

  // Compute stress from strain energy using automatic differentiation
  auto P_stress_AD = get_gradient(free_energy);

  // Get difference
  auto stress_difference = P_stress - P_stress_AD;

  // Check that the stress is consistent with the strain energy
  EXPECT_LT(norm(stress_difference), 1e-8);

  // Note: The part below is just to make sure that AD is doing it's thing correctly
  // Check that the AD-computed derivatives of energy w.r.t. displacement_gradient 
  // are in agreement with results obtained from a central finite difference stencil
  {
    double epsilon = 1.0e-6;
    tensor<double, 3, 3> perturbation{{{0.1, 0.0, 0.6}, 
                                      {1.0, 0.4, 0.5}, 
                                      {0.3, 0.2, 0.1}}};

    auto energy1 = LCEMat_Bertoldi.calculateStrainEnergy(
      state, H - epsilon * perturbation, order_param_tuple, gamma_param_tuple);

    auto energy2 = LCEMat_Bertoldi.calculateStrainEnergy(
      state, H + epsilon * perturbation, order_param_tuple, gamma_param_tuple);

    auto denergy = get_gradient(LCEMat_Bertoldi.calculateStrainEnergy(
      state, make_dual(H), order_param_tuple, gamma_param_tuple));

    auto error = double_dot(denergy, perturbation) - (energy2 - energy1) / (2 * epsilon);

    EXPECT_NEAR(error / double_dot(denergy, perturbation), 0.0, 1e-8); 
  }
}

// --------------------------------------------------------

TEST(TestLiquidCrystalBertoldiMat, AgreesWithIsotropicInOrderParameterLimitAndSmallStrain)
{
  double density = 1.0;
  double young_modulus = 1.0;
  double possion_ratio = 0.49;
  double beta_param = 1.0;
  double max_order_param = 0.5;

  tuple <double, int> order_param_tuple, gamma_param_tuple;
  order_param_tuple = make_tuple(max_order_param, 123); // same as initial to remove its contribution
  gamma_param_tuple = make_tuple(0.6, 456);

  LiqCrystElast_Bertoldi LCEMat_Bertoldi(density, young_modulus, possion_ratio, max_order_param, beta_param);

  // test conditions
  tensor<double, 3, 3> H{{{0.000123,  0.00011, -0.000123},
                          {-0.00019, 0.00015, 0.0},
                          {0.0,  0.00011 , 0.00019}}};

  // liquid crystal elastomer model response
  LiqCrystElast_Bertoldi::State state{};
  auto stress_Bertoldi = LCEMat_Bertoldi(state, H, order_param_tuple, gamma_param_tuple);

  // neo-hookean for comparison
  double bulk_modulus = young_modulus / 3.0 / (1.0 - 2.0*possion_ratio);
  double shear_modulus = 0.5 * young_modulus / (1.0 + possion_ratio);

  solid_mechanics::LinearIsotropic solidMat_isotropic{.density = density, .K = bulk_modulus, .G = shear_modulus};
  solid_mechanics::LinearIsotropic::State state_isotropic{};
  auto stress_isotropic = solidMat_isotropic(state_isotropic, H);

  // Get difference
  auto stress_difference = stress_Bertoldi - stress_isotropic;

  // Check that the stress is consistent with the strain energy
  EXPECT_NEAR(norm(stress_difference), 0.0, 1e-5);
}

// --------------------------------------------------------

TEST(TestLiquidCrystalBertoldiMat, IdentityDefGradientNoEnergyNorStress)
{
  double density = 1.0;
  double young_modulus = 1.0;
  double possion_ratio = 0.49;
  double beta_param = 1.0;
  double max_order_param = 0.5;

  tuple <double, int> order_param_tuple, gamma_param_tuple;
  order_param_tuple = make_tuple(max_order_param, 123); // same as initial to remove its contribution
  gamma_param_tuple = make_tuple(0.6, 456);

  LiqCrystElast_Bertoldi LCEMat_Bertoldi(density, young_modulus, possion_ratio, max_order_param, beta_param);

  // test conditions
  tensor<double, 3, 3> H{{{0.0,  0.0, 0.0},
                          {0.0, 0.0, 0.0},
                          {0.0,  0.0, 0.0}}};

  // liquid crystal elastomer model response
  LiqCrystElast_Bertoldi::State state{};
  auto stress_Bertoldi = LCEMat_Bertoldi(state, H, order_param_tuple, gamma_param_tuple);

  // Strain energy
  auto free_energy = LCEMat_Bertoldi.calculateStrainEnergy(state, H, order_param_tuple, gamma_param_tuple);

  // Check that the stress is consistent with the strain energy
  EXPECT_LT(free_energy, 1e-8);
  EXPECT_LT(norm(stress_Bertoldi), 1e-8);
}

// --------------------------------------------------------

TEST(TestLiquidCrystalBertoldiMat, orderParameterSweep)
{
  double density = 1.0;
  double young_modulus = 1.0;
  double possion_ratio = 0.49;
  double beta_param = 1.0;
  double max_order_param = 0.5;                                                                                                                                                                                          

  LiqCrystElast_Bertoldi LCEMat_Bertoldi(density, young_modulus, possion_ratio, max_order_param, beta_param);

  // liquid crystal elastomer model response
  LiqCrystElast_Bertoldi::State state{};

  unsigned int num_steps = 5;
  double max_time = 1.0;
  double dt = max_time / num_steps;
  double t = 0;

  // Note: implemented here something similar to what is in the uniaxial_stress_test methdo in the material_verification_tools.hpp file
  // since I was not sure how to explicitly change the parameters provided as doubles in a tuple.

  // strain function of time
  double strain_rate = 1e-2;
  std::function<double(double)> epsilon_xx = [strain_rate](double time){ return strain_rate + 0.0*strain_rate*time; }; // Kept constant here

  auto sigma_yy_and_zz = [&epsilon_xx, &max_order_param, &max_time, &t, LCEMat_Bertoldi, state](auto x) 
  {
    auto epsilon_yy = x[0];
    auto epsilon_zz = x[1];
    using T         = decltype(epsilon_yy);
    tensor<T, 3, 3> du_dx{};

    du_dx[0][0] = epsilon_xx(t);
    du_dx[1][1] = epsilon_yy;
    du_dx[2][2] = epsilon_zz;

    auto copy   = state;
    tuple <double, int> copy_order_param_tuple, copy_gamma_param_tuple;
    copy_order_param_tuple = make_tuple(max_order_param* t / max_time, 123);
    copy_gamma_param_tuple = make_tuple(0.6, 456);
    auto stress = LCEMat_Bertoldi(copy, du_dx, copy_order_param_tuple, copy_gamma_param_tuple);

    return tensor{{stress[1][1], stress[2][2]}};
  };

  // Loop over time and change order parameter from min to max
  tensor<double, 3, 3> dudx{};
  tensor<double, 5> refSxx{0.432068, 0.339607, .252902, 0.170087, 0.0896447};

  if(num_steps!=5){std::cout<<"... Test temporary hardcoded for 5 steps because of refSxx definition"<<std::endl; exit(0);}
  bool printStress(false);

  for (size_t i = 0; i < num_steps; i++) 
  {
    auto initial_guess     = tensor<double, 2>{dudx[1][1], dudx[2][2]};
    auto epsilon_yy_and_zz = find_root(sigma_yy_and_zz, initial_guess);
    dudx[0][0]             = epsilon_xx(t);
    dudx[1][1]             = epsilon_yy_and_zz[0];
    dudx[2][2]             = epsilon_yy_and_zz[1];

    tuple <double, int> order_param_tuple, gamma_param_tuple;
    order_param_tuple = make_tuple(max_order_param* t / max_time, 123);
    gamma_param_tuple = make_tuple(0.6, 456);
    auto stress = LCEMat_Bertoldi(state, dudx, order_param_tuple, gamma_param_tuple);

    t += dt;
    
    EXPECT_NEAR(stress[0][0], refSxx[static_cast<int>(i)], 1e-6);

    if(printStress)
    {
      std::cout << "... Sxx = " << stress[0][0] << ", Syy = " << stress[1][1] << ", Szz = " << stress[2][2] << std::endl;
    }
  }
}

// // --------------------------------------------------------

// TEST(TestLiquidCrystalBertoldiMat, strainAndOrderParamSweep)
// {
//   double density = 1.0;
//   double nu = 0.48;
//   double shear_modulus = 13.33e3;
//   double E = 2.0 * (1.0 + nu) * shear_modulus;
//   double bulk_modulus = E / 3.0 / (1.0 - 2.0*nu);
//   double order_constant = 10.0;
//   double order_parameter = 1.0;
//   double initial_temperature = 300.0;
//   double transition_temperature = 370.0;
//   double max_temperature = 400.0;
//   tensor<double, 3> normal{{0.0, 1.0, 0.0}};
//   double Nb2 = 1.0;
  
//   LiqCrystElast_Bertoldi LCEMat_Bertoldi(density, shear_modulus, bulk_modulus, order_constant, order_parameter, transition_temperature, normal, Nb2);

//   auto initial_distribution = LiqCrystElast_Bertoldi::calculateInitialDistributionTensor(normal, order_parameter, Nb2);
//   decltype(LCEMat_Bertoldi)::State initial_state{DenseIdentity<3>(), initial_distribution, initial_temperature, order_parameter};
//   double max_time = 1.0;
//   unsigned int steps = 20;

//   double strain_rate = 2e-1;
//   std::function<double(double)> strain_rate_func = [strain_rate](double t){ 
//         if(t<0.25)
//         {
//           return strain_rate*4*t;
//         }
//         else if(t>=0.25 && t<0.5)
//         {
//           return strain_rate;
//         }
//         else if(t>=0.5 && t<0.75)
//         {
//           return strain_rate*4*(0.75-t); 
//         }
//         else
//         {
//           return 0.0;
//         }
//     };
    
//   std::function<double(double)> temperature_func =
//       [initial_temperature, max_temperature](double t) {
//         if(t<0.25)
//         {
//           return initial_temperature;
//         }
//         else if(t>=0.25 && t<0.5)
//         {
//           return initial_temperature + 4*(t-0.25)*(max_temperature - initial_temperature);
//         }
//         else if(t>=0.5 && t<0.75)
//         {
//           return max_temperature;
//         }
//         else
//         {
//           return max_temperature - 4*(t-0.75)*(max_temperature - initial_temperature);
//         }
//       };

//   auto response_history = uniaxial_stress_test(max_time, steps, LCEMat_Bertoldi, initial_state, strain_rate_func, temperature_func);

//   bool printOutput(false);
  
//   for (unsigned int i = 0; i < steps; i++) 
//   {
//     auto [t, strain, stress, state] = response_history[i];

//     if(printOutput)
//     {
//       std::cout << "... Time: " << t
//                 << ", q: " << state.order_parameter
//                 << ", e_xx: " << strain[0][0]
//                 << ", e_yy: " << strain[1][1]
//                 << ", e_zz: " << strain[2][2]
//                 << ", Order parameter: " << state.instantaneous_order_parameter
//                 << ", sigma_xx: " << stress[0][0]
//                 << std::endl;
//     }
//   }
// }

// --------------------------------------------------------

} // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();

  return result;
}

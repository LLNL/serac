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

TEST(TestLiquidCrystalMaterial, ConsistentStressDerivedFromStrainEnergy)
{
  double density = 1.0;
  double young_modulus = 1.0;
  double possion_ratio = 0.49;
  double beta_param = 1.0;
  double initial_order_param = 0.5;

  tuple <double, int> order_param_tuple, gamma_param_tuple;
  order_param_tuple = make_tuple(0.1, 123);
  gamma_param_tuple = make_tuple(0.9, 456);

  LiqCrystElast_Bertoldi material(density, young_modulus, possion_ratio, initial_order_param, beta_param);

  // test conditions
  tensor<double, 3, 3> H{{{0.423,  0.11, -0.123},
                          {-0.19, 0.25, 0.0},
                          {0.0,  0.11 , 0.39}}};

  // liquid crystal elastomer model response
  LiqCrystElast_Bertoldi::State state{};
  auto stress = material(state, H, order_param_tuple, gamma_param_tuple);

  // Transform Cauchy stress into Piola stress, which is what the AD computation returns
  auto I = Identity<3>();  
  auto F = H + I;
  auto J = det(F);
  auto P_stress = J*stress*inv(transpose(F));

  // Strain energy of the model
  auto free_energy = material.calculateStrainEnergy(state, make_dual(H), order_param_tuple, gamma_param_tuple);

  // Compute stress from strain energy using automatic differentiation
  auto P_stress_AD = get_gradient(free_energy);

  // Get difference
  auto stress_difference = P_stress - P_stress_AD;

  // Check that the stress is consistent with the strain energy
  EXPECT_LT(norm(stress_difference), 1e-8);
}

// --------------------------------------------------------

// TEST(TestLiquidCrystalMaterial, AgreesWithNeoHookeanInOrderParameterLimit)
// {
//   double density = 1.0;
//   double E = 1.0;
//   double possion_ratio = 0.49;
//   double order_constant = 1.0;
//   double order_parameter = 0.0;
//   double transition_temperature = 1.0;
//   tensor<double, 3> normal{{1.0, 0.0, 0.0}};
//   double Nb2 = 1.0;
//   double shear_modulus = 0.5*E/(1.0 + possion_ratio);
//   double bulk_modulus = E / 3.0 / (1.0 - 2.0*possion_ratio);  

// 
// std::cout << "\n..... P_stress ....." << std::endl;
// std::cout << P_stress << std::endl;
// std::cout << "\n..... P_stress_AD ....." << std::endl;
// std::cout << P_stress_AD << std::endl;
// std::cout << "\n..... stress_difference ....." << std::endl;
// std::cout << stress_difference << std::endl;
// std::cout << std::endl;
// //
//   LiqCrystElast_Brighenti_Material material(density, bulk_modulus, possion_ratio, order_constant,
//                                order_parameter, transition_temperature, normal, Nb2);

//   // test conditions
//   tensor<double, 3, 3> H{{{-0.474440607694436,  0.109876281988692, -0.752574057841232},
//                           {-0.890004651428391, -1.254064550255045, -0.742440671831607},
//                           {-0.310665550306666,  0.90643674423369 , -1.090724491652343}}};
//   double theta = 300.0; // far above transition temperature

  // double epsilon = 1.0e-6;
  // tensor<double, 3, 3> perturbation{{{0.1, 0.0, 0.6}, 
  //                                    {1.0, 0.4, 0.5}, 
  //                                    {0.3, 0.2, 0.1}}};

  // // Check that the AD-computed derivatives of energy w.r.t. displacement_gradient 
  // // are in agreement with results obtained from a central finite difference stencil
  // {
  //   auto energy1 = material.calculateStrainEnergy(
  //     state, H - epsilon * perturbation, order_param_tuple, gamma_param_tuple);

  //   auto energy2 = material.calculateStrainEnergy(
  //     state, H + epsilon * perturbation, order_param_tuple, gamma_param_tuple);

  //   auto denergy = get_gradient(material.calculateStrainEnergy(
  //     state, make_dual(H), order_param_tuple, gamma_param_tuple));

  //   auto error = double_dot(denergy, perturbation) - (energy2 - energy1) / (2 * epsilon);

  //   EXPECT_NEAR(error / double_dot(denergy, perturbation), 0.0, 1e-8); 
  // }


//   // liquid crystal elastomer model response
//   auto F_old = DenseIdentity<3>();
//   double theta_old = 300.0;
//   tensor<double, 3, 3> mu_old = LiqCrystElast_Brighenti_Material::calculateInitialDistributionTensor(normal, order_parameter, Nb2);
//   LiqCrystElast_Brighenti_Material::State state{F_old, mu_old, theta_old, order_parameter};
//   auto response = material(state, H, theta);

//   // neo-hookean for comparison
//   solid_mechanics::NeoHookean nh_material{.density = density, .K = bulk_modulus, .G = shear_modulus};
//   solid_mechanics::NeoHookean::State nh_state{};
//   auto nh_response = nh_material(nh_state, H);

//   auto stress_difference = response - nh_response;
//   EXPECT_LT(norm(stress_difference), 1e-8);
// }

// --------------------------------------------------------

// TEST(TestLiquidCrystalMaterial, agreesWithNeoHookeanInOrderParameterLimitOverEntireUniaxialTest)
// {
// return;
//   double density = 1.0;
//   double E = 1.0;
//   double nu = 0.25;
//   double shear_modulus = 0.5*E/(1.0 + nu);
//   double bulk_modulus = E / 3.0 / (1.0 - 2.0*nu);
//   double order_constant = 1.0;
//   double order_parameter = 1.0e-10;
//   double transition_temperature = 1.0;
//   tensor<double, 3> normal{{0.0, 1.0, 0.0}};
//   double Nb2 = 1.0;
  
//   LiqCrystElast_Bertoldi material(density, shear_modulus, bulk_modulus, order_constant, order_parameter, transition_temperature, normal, Nb2);
//   double temperature = 300.0; // far above transition temperature

//   auto initial_distribution = LiqCrystElast_Bertoldi::calculateInitialDistributionTensor(normal, order_parameter, Nb2);
//   decltype(material)::State initial_state{DenseIdentity<3>(), initial_distribution, temperature, order_parameter};
//   double max_time = 20.0;
//   unsigned int steps = 10;
//   double strain_rate = 1e-2;
//   std::function<double(double)> constant_strain_rate = [strain_rate](double t){ return strain_rate*t; };
//   std::function<double(double)> constant_temperature = [temperature](double){ return temperature; };
//   auto response_history = uniaxial_stress_test(max_time, steps, material, initial_state, constant_strain_rate, constant_temperature);

//   solid_mechanics::NeoHookean nh_material{.density = density, .K = bulk_modulus, .G = shear_modulus};
//   solid_mechanics::NeoHookean::State nh_initial_state{};
//   auto nh_response_history = uniaxial_stress_test(max_time, steps, nh_material, nh_initial_state, constant_strain_rate);

//   for (size_t i = 0; i < steps; i++) {
//     auto [t, strain, stress, state] = response_history[i];
//     auto [nh_t, nh_strain, nh_stress, nh_state_loop] = nh_response_history[i];
//     double difference = std::abs(stress[0][0] - nh_stress[0][0]);
//     EXPECT_LT(difference, 1e-8);

//     std::cout << "+++ Time: " << t
//               << " , strain: " << strain
//               << " , order parameter: " << state.instantaneous_order_parameter
//               << " , stress_xx: " << stress[0][0] << std::endl;
//   }
// }

// // --------------------------------------------------------

// TEST(TestLiquidCrystalMaterial, orderParameterSweep)
// {
// return;
//   double density = 1.0;
//   double E = 1.0;
//   double nu = 0.25;
//   double shear_modulus = 0.5*E/(1.0 + nu);
//   double bulk_modulus = E / 3.0 / (1.0 - 2.0*nu);
//   double order_constant = 1.0;
//   double order_parameter = 1.0;
//   double transition_temperature = 10.0;
//   tensor<double, 3> normal{{0.0, 1.0, 0.0}};
//   double Nb2 = 1.0;
  
//   LiqCrystElast_Bertoldi material(density, shear_modulus, bulk_modulus, order_constant, order_parameter, transition_temperature, normal, Nb2);
//   double initial_temperature = 5.0;

//   auto initial_distribution = LiqCrystElast_Bertoldi::calculateInitialDistributionTensor(normal, order_parameter, Nb2);
//   decltype(material)::State state{DenseIdentity<3>(), initial_distribution, initial_temperature, order_parameter};
//   double max_time = 1.0;
//   unsigned int steps = 50;
//   double time = 0;
//   double dt = max_time / steps;
//   tensor<double, 3, 3> H{};
//   std::function<double(double)> temperature_func =
//       [initial_temperature, transition_temperature](double t) {
//         return initial_temperature + 2*t*(transition_temperature - initial_temperature);
//       };

//   for (unsigned int i = 0; i < steps; i++) {
//     time += dt;
//     double temperature = temperature_func(time);
//     material(state, H, temperature);
//     std::cout << state.distribution_tensor[1][1] << std::endl;
//   }
// }

// // --------------------------------------------------------

// TEST(TestLiquidCrystalMaterial, isNotDegenerate)
// {
//   // This is a dummy test that should be eventually removed. I (BT) am
//   // adding it only to demonstrate something unusual about this
//   // model's behavior. This is not an aspect of the model we actually
//   // want to enforce, it's just an unfortunate behavior built into
//   // this model.
//   double density = 1.0;
//   double E = 1.0;
//   double nu = 0.25;
//   double shear_modulus = 0.5*E/(1.0 + nu);
//   double bulk_modulus = E / 3.0 / (1.0 - 2.0*nu);
//   double order_constant = 1.0;
//   double order_parameter = 1.0; // this is the culprit. This must be less than 1 for the model to give meaningful results.
//   double transition_temperature = 10.0;
//   tensor<double, 3> normal{{0.0, 1.0, 0.0}};
//   double Nb2 = 1.0;

//   LiqCrystElast_Bertoldi material(density, shear_modulus, bulk_modulus, order_constant, order_parameter, transition_temperature, normal, Nb2);
//   double initial_temperature = 5.0;

//   auto initial_distribution = LiqCrystElast_Bertoldi::calculateInitialDistributionTensor(normal, order_parameter, Nb2);
//   decltype(material)::State state{DenseIdentity<3>(), initial_distribution, initial_temperature, order_parameter};
//   tensor<double, 3, 3> H{};
//   auto F = DenseIdentity<3>();
//   double dlambda = 0.1;
//   const unsigned int steps = 10;

//   // It is easy to verify that when this model is fully ordered in the
//   // reference state (which means `order_parameter` is set to 1), ANY
//   // diagonal deformation gradient with F[1][1] = 1.0 will cause no
//   // stress in the deviatoric response*. For this example, I assign an
//   // increasing sequence of F[0][0] and keep F[1][1] = 1.0. Then, all
//   // one needs to do to have a zero stress state is to keep the volume
//   // fixed, that is, sef F[2][2] = 1/[F[0][0]. No matter how large the
//   // tensile F[0][0] deformation is, the model will stay at zero
//   // stress. This is BAD - the model cannot be used in this condition,
//   // because the field problem will not have a solution. We should put
//   // a check in the model so that it errors if the initial order is
//   // close to 1.
//   //
//   // * This assumed that the normal vector is set as {0.0, 1.0,
//   // 0.0}. If the vector points in another direction, the degeneracy
//   // will involve more components of the deformation gradient. The
//   // conclusions about the model problems are the same.
//   for (unsigned int i = 0; i < steps; i++) {
//     F[0][0] += dlambda;
//     F[2][2] = 1.0/F[0][0];
//     H = F - DenseIdentity<3>();
//     auto response = material(state, H, initial_temperature);
//     EXPECT_LT(response[0][0], 1e-8);
//   }
// }

// // --------------------------------------------------------

// TEST(TestLiquidCrystalMaterial, strainAndOrderParamSweep)
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
  
//   LiqCrystElast_Bertoldi material(density, shear_modulus, bulk_modulus, order_constant, order_parameter, transition_temperature, normal, Nb2);

//   auto initial_distribution = LiqCrystElast_Bertoldi::calculateInitialDistributionTensor(normal, order_parameter, Nb2);
//   decltype(material)::State initial_state{DenseIdentity<3>(), initial_distribution, initial_temperature, order_parameter};
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

//   auto response_history = uniaxial_stress_test(max_time, steps, material, initial_state, strain_rate_func, temperature_func);

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

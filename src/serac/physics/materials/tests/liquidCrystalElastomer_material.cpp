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

#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/tuple.hpp"
#include "serac/numerics/functional/tuple_arithmetic.hpp"

namespace serac
{

static constexpr auto I = Identity<3>();

/**
 * @brief Compute Green's strain from the displacement gradient
 */
template <typename T>
auto tensorSquareRoot(const tensor<T, 3, 3>& A)
{
  auto X = A;
  for (int i = 0; i < 15; i++) {
    X = 0.5 * (X + dot(A, inv(X)));
  }
  return X;
}

/// ---------------------------------------------------------------------------

/// @brief Green-Saint Venant isotropic thermoelastic model
struct LCEMaterialProperties
{
  double Gshear;          ///< shear modulus (=ca*KB*T) [Pa]
  // double kB;         ///< Boltzmann constant [J/K]
  // double ca;         ///< number of mechanically active chains [1/m^3]
  double N_seg;      ///< rigid (Khun’s) segments in chain
  double b;          ///< segment length
  double T_ni;       ///< nematic-isotropic transition temperature
  double c;          ///< material-specific constant parameter
  double q0;         // initial nematic order parameter
  double p;           // hydrostatic pressure
  struct State { /* this material has no internal variables */ };

/// ---------------------------------------------------------------------------

  /**
   * @brief Evaluate constitutive variables for thermomechanics
   *
   * @param[in] grad_u displacement gradient
   * @param[in] theta temperature
   * @param[in] grad_theta temperature gradient
   * @param[in] grad_u_old displacement gradient at previous time step
   * @param[in] theta_old temperature at previous time step
   * @param[in] dt time increment
   * @param[out] P Piola stress
   * @param[out] cv0 volumetric heat capacity in ref config
   * @param[out] s0 internal heat supply in ref config
   * @param[out] q0 Piola heat flux
   */
  template <typename T1, typename T2, typename T3>
  auto calculateConstitutiveOutputs(
    const tensor<T1, 3, 3>& grad_u, 
    T2 theta, 
    const tensor<T3, 3>& grad_theta, 
    State& /*state*/, 
    const tensor<double, 3, 3>& grad_u_old, 
    double theta_old,
    double dt)
  {
    tensor<double, 1, 3> normal = {2/std::sqrt(3), -1/std::sqrt(3), 3/std::sqrt(3)};
    normal = normal/norm(normal);

    // Deformation gradients
    auto         F     = grad_u + I;
    auto         F_old = grad_u_old + I;
    auto         F_hat = F * inv(F_old);

    // Polar decomposition of deformation gradient based on F_hat
    auto U_hat = tensorSquareRoot(transpose(F_hat) * F_hat);
    auto R_hat = F_hat * inv (U_hat);

    // Determinant of deformation gradient
    auto J = det(F_hat);

    // Nematic order scalar
    double q_old = q0 / (1 + std::exp((theta_old - T_ni)/c));
    double q     = q0 / (1 + std::exp((theta - T_ni)/c));

    // Nematic order tensor
    auto Q_old = q_old/2 * (3 * transpose(normal) * normal - I);
    auto Q     = q/2 * (3 * transpose(normal) * normal - I);

    // Initial distribution tensor
    auto mu_0_a = (1 - q0) * I;
    auto mu_0_b = 3 * q0 * (transpose(normal) * normal);
    auto mu_0 = N_seg*std::pow(b,2)/3 * (mu_0_a + mu_0_b);
    
    // Distribution tensor (using 'Strang Splitting' approach)
    auto mu_old_a = (1 - q_old) * I;
    auto mu_old_b = 3 * q_old * (transpose(normal) * normal);
    auto mu_old   = N_seg*std::pow(b,2)/3 * (mu_old_a + mu_old_b);

    auto mu_a = F_hat * ( mu_old + 2*N_seg*std::pow(b,2)/3* (Q - Q_old)) * transpose(F_hat);
    auto mu_b = 2*N_seg*std::pow(b,2)/3 * (Q - R_hat * Q * transpose(R_hat));
    auto mu   = mu_a + mu_b;

    // stress
    const auto P =  J * ( (3*Gshear/(N_seg*std::pow(b,2))) * (mu - mu_0) + p*I ) * inv(transpose(F_hat));

    // internal heat source
    const double s0 = -3 * N_seg * theta * tr(F_hat)/dt;

    // heat flux
    const auto f0 = -N_seg * grad_theta;

    return serac::tuple{P, mu, s0, f0};
  }

/// ---------------------------------------------------------------------------

  /**
   * @brief evaluate free energy density
   */
  template <typename T1, typename T2>
  auto calculateFreeEnergy(
    const tensor<T1, 3, 3>& grad_u,
    const tensor<double, 3, 3>& grad_u_old,
    T2 theta,
    T2 theta_old)
  {
    tensor<double, 1, 3> normal = {2/std::sqrt(3), -1/std::sqrt(3), 3/std::sqrt(3)};
    normal = normal/norm(normal);

    // Deformation gradients
    auto         F     = grad_u + I;
    auto         F_old = grad_u_old + I;
    auto         F_hat = F * inv(F_old);

    // Polar decomposition of deformation gradient based on F_hat
    auto U_hat = tensorSquareRoot(transpose(F_hat) * F_hat);
    auto R_hat = F_hat * inv (U_hat);

    // Determinant of deformation gradient
    auto J = det(F_hat);

    // Nematic order scalar
    double q_old = q0 / (1 + std::exp((theta_old - T_ni)/c));
    double q     = q0 / (1 + std::exp((theta - T_ni)/c));

    // Nematic order tensor
    auto Q_old = q_old/2 * (3 * transpose(normal) * normal - I);
    auto Q     = q/2 * (3 * transpose(normal) * normal - I);

    // Initial distribution tensor
    auto mu_0_a = (1 - q0) * I;
    auto mu_0_b = 3 * q0 * (transpose(normal) * normal);
    auto mu_0 = N_seg*std::pow(b,2)/3 * (mu_0_a + mu_0_b);
    
    // Distribution tensor (using 'Strang Splitting' approach)
    auto mu_old_a = (1 - q_old) * I;
    auto mu_old_b = 3 * q_old * (transpose(normal) * normal);
    auto mu_old   = N_seg*std::pow(b,2)/3 * (mu_old_a + mu_old_b);

    auto mu_a = F_hat * ( mu_old + 2*N_seg*std::pow(b,2)/3* (Q - Q_old)) * transpose(F_hat);
    auto mu_b =  2*N_seg*std::pow(b,2)/3 * (Q - R_hat * Q * transpose(R_hat));
    auto mu   = mu_a + mu_b;

    auto psi_1 = 3 * Gshear/(2*N_seg*std::pow(b,2)) * tr(mu - mu_0);
    auto psi_2 = p * (J - 1);

    return psi_1 + psi_2;
  }

/// ---------------------------------------------------------------------------

  /**
   * @brief Return constitutive output for thermal operator
   */
  auto calculateThermalConstitutiveOutputs(
    const tensor<double, 3, 3>& grad_u, 
    double theta,
    const tensor<double, 3>& grad_theta, 
    State& state,
    const tensor<double, 3, 3>& grad_u_old, 
    double theta_old, 
    double dt)
  {
    auto [P, mu0, s0, f0] = calculateConstitutiveOutputs(grad_u, theta, grad_theta, state, grad_u_old, theta_old, dt);
    return serac::tuple{mu0, s0, f0};
  }

/// ---------------------------------------------------------------------------

  /**
   * @brief Return constitutive output for mechanics operator
   */
  auto calculateMechanicalConstitutiveOutputs(
    const tensor<double, 3, 3>& grad_u, 
    double theta,
    const tensor<double, 3>& grad_theta, 
    State& state,
    const tensor<double, 3, 3>& grad_u_old, 
    double theta_old, 
    double dt)
  {
    auto [P, mu0, s0, f0] = calculateConstitutiveOutputs(grad_u, theta, grad_theta, state, grad_u_old, theta_old, dt);
    return P;
  }
};

// --------------------------------------------------------

// The energy of the polymer is non-zero in its stress-free state.
TEST(LiqCrystElastMaterial, FreeEnergyIsZeroInReferenceState)
{
return;
  LCEMaterialProperties material{
    .Gshear = 13.33e3,
    .N_seg = 1.0,
    .b     = 1.0,         
    .T_ni  = 273+92,      
    .c     = 10,      
    .q0    = 0.46,
    .p     = 30};
  tensor<double, 3, 3>  displacement_grad{};
  tensor<double, 3, 3>  displacement_grad_old{};
  double                temperature = 300.0;
  double                temperature_old = temperature+2;
  double                free_energy = material.calculateFreeEnergy(displacement_grad, displacement_grad_old, temperature, temperature_old);

  EXPECT_NEAR(free_energy, 0.0, 1e-10);
}

// --------------------------------------------------------

// The energy of the polymer is non-zero in its stress-free state.
TEST(LiqCrystElastMaterial, StressIsNonZeroInReferenceState)
{
return;  
  LCEMaterialProperties material{
    .Gshear = 13.33e3,
    .N_seg = 1.0,
    .b     = 1.0,         
    .T_ni  = 273+92,      
    .c     = 10,      
    .q0    = 0.46,
    .p     = 30};
  tensor<double, 3, 3>  displacement_grad{};
  double                temperature = 300;
  tensor<double, 3>     temperature_grad{};
  LCEMaterialProperties::State state{};
  auto                         displacement_grad_old = displacement_grad;
  double                       temperature_old       = temperature;
  double                       dt                    = 1.0;
  auto stress = material.calculateMechanicalConstitutiveOutputs(displacement_grad, temperature, temperature_grad, state,
                                                                displacement_grad_old, temperature_old, dt);

  EXPECT_NEAR(norm(stress), 56.6426, 1e-4);
}

// --------------------------------------------------------

TEST(LiqCrystElastMaterial, FreeEnergyAndStressAgree)
{
  LCEMaterialProperties material{
    .Gshear = 13.33e3,
    .N_seg = 1.0,
    .b     = 1.0,         
    .T_ni  = 273+92,      
    .c     = 10,      
    .q0    = 0.46,
    .p     = 30};
  // clang-format off
  tensor<double, 3, 3>  displacement_grad_old{{{0.0, 0.133333333333333, 0.0},
                                           {0.0, -0.019933422158758, 0.198669330795061},
                                           {0.0, -0.198669330795061, -0.019933422158758}}};
  tensor<double, 3, 3> displacement_grad{{{0.0, 0.266666666666667, 0.0},
                                           {0.0, -0.078939005997115, 0.389418342308651},
                                           {0.0, -0.389418342308651, -0.078939005997115}}};
  // clang-format on
  double                       temperature = 321.4141;
  double                       temperature_old = 320.7035;
  tensor<double, 3>            temperature_grad{0.87241435, 0.11105156, -0.27708054};
  
  LCEMaterialProperties::State state{};
  double dt = 0.2;

  auto energy_and_stress = material.calculateFreeEnergy(
    make_dual(displacement_grad), displacement_grad_old, temperature, temperature_old);
  auto stress = material.calculateMechanicalConstitutiveOutputs(
    displacement_grad, temperature, temperature_grad, state, displacement_grad_old, temperature_old, dt);

  auto stress_AD = get_gradient(energy_and_stress);

std::cout<<std::endl;
print(stress);
std::cout<<std::endl;
print(stress_AD);
std::cout<<std::endl;

  auto error  = stress - stress_AD;

  EXPECT_NEAR(norm(error), 0.0, 1e-12);
}
}  // namespace serac

// --------------------------------------------------------

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();

  return result;
}

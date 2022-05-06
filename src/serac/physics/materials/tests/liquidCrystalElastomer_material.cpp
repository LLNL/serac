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

namespace serac {

static constexpr auto I = Identity<3>();

/**
 * @brief Compute Green's strain from the displacement gradient
 */
template <typename T>
auto greenStrain(const tensor<T, 3, 3>& grad_u)
{
  return 0.5 * (grad_u + transpose(grad_u) + dot(transpose(grad_u), grad_u));
}

/// @brief Green-Saint Venant isotropic thermoelastic model
struct LCEMaterialProperties {

double E;          ///< Young's modulus
double nu;         ///< Poisson's ratio
double C;          ///< volumetric heat capacity
double alpha;      ///< thermal expansion coefficient
double theta_ref;  ///< datum temperature
double k;          ///< thermal conductivity

  // double G;          ///< shear modulus (=ca*KB*T) [Pa]
  // double kB;         ///< Boltzmann constant [J/K]
  // double ca;         ///< number of mechanically active chains [1/m^3]
  // double N_seg;      ///< rigid (Khun’s) segments in chain
  // double b;          ///< segment length
  // double T_ni;       ///< nematic-isotropic transition temperature
  // double c;          ///< material-specific constant parameter

  struct State { /* this material has no internal variables */
  };

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

const double K    = E / (3.0 * (1.0 - 2.0 * nu));
const double G    = 0.5 * E / (1.0 + nu);
const auto   Eg   = greenStrain(grad_u);
const auto   trEg = tr(Eg);

    // const double ca    = 3.2183e+24;   // number of mechanically active chains [1/m^3]
    // const double kB    = 1.380649e-23; // Boltzmann constant [J/K]
    const double Gshear     = 13.33e3;      // shear modulus (=ca*KB*T) [Pa]
    const double N_seg = 1.0;          // rigid (Khun’s) segments in chain
    const double b     = 1.0;          // segment length
    const double T_ni  = 273+92;       // nematic-isotropic transition temperature
    const double c     = 10;           // material-specific constant parameter
    const double q0    = 0.46;         // initial nematic order parameter
    const double p     = 30;           // hydrostatic pressure
    tensor<double, 1, 3> normal = {2/std::sqrt(3), -1/std::sqrt(3), 3/std::sqrt(3)};

    // Deformation gradients
    auto         F     = grad_u + I;
    auto         F_old = grad_u_old + I;
    auto         F_hat = F * inv(F_old);

    // Polar decomposition of deformation gradient based on F_hat
    auto U_hat = transpose(F_hat) * F_hat; // need to do pow(.,0.5) still
    auto R_hat = F_hat * inv (U_hat);

    // Velocity gradient using Hughes-Winget approx
    // auto L = 2.0 / dt * (F_hat - I) * inv(F_hat + I);

    // Spin tensor
    // auto W = 0.5 * (L - transpose(L));

    // Determinant of deformation gradient
    auto J = det(F_hat);

    // Nematic order scalar
    double q_old = q0 / (1 + std::exp((theta_old - T_ni)/c));
    double q     = q0 / (1 + std::exp((theta - T_ni)/c));

    // Nematic order tensor
    auto Q_old = q_old/2 * (3 * transpose(normal) * normal - I);
    auto Q     = q/2 * (3 * transpose(normal) * normal - I);

    // Distribution tensor (using 'Strang Splitting' approach)
    auto mu0_a = I - Q_old ;
    auto mu0_b = 3 * Q_old * (transpose(normal) * normal);
    auto mu_0 = N_seg*std::pow(b,2/3) * (mu0_a + mu0_b);

    auto mu_a = F_hat * ( mu_0 + 2*N_seg*std::pow(b,2/3)* (Q - Q_old)) * transpose(F_hat);
    auto mu_b =  2*N_seg*std::pow(b,2/3) * (Q - R_hat * Q * transpose(R_hat));
    auto mu = mu_a + mu_b;

    // stress
    const auto S = I + 0.0 * G * dev(Eg) + K * (trEg - 3.0 * alpha * (theta - theta_ref)) * I;
    // const auto P = J * mu * W * R_hat * S;
    const auto P =  S * J * ( (3*Gshear/(N_seg*b*b)) * (mu - mu_0) + p*I ) * inv(transpose(F_hat));

    // internal heat source
    // use backward difference to estimate rate of green strain
    const auto   Eg_old = greenStrain(grad_u_old);
    const double egdot  = (trEg - tr(Eg_old)) / dt;
    const double s0     = -3 * K * alpha * theta * egdot;

    // heat flux
    const auto q0_ = -k * grad_theta;

    return serac::tuple{P, C, s0, q0_};
  }

  /**
   * @brief Return constitutive output for thermal operator
   *
   * @param[in] grad_u displacement gradient
   * @param[in] theta temperature
   * @param[in] grad_theta temperature gradient
   * @param[in] grad_u_old displacement gradient at previous time step
   * @param[in] theta_old temperature at previous time step
   * @param[in] dt time increment
   * @param[out] cv0 volumetric heat capacity in ref config
   * @param[out] s0 internal heat supply in ref config
   * @param[out] q0 Piola heat flux
   */
  auto calculateThermalConstitutiveOutputs(const tensor<double, 3, 3>& grad_u, double theta,
                                           const tensor<double, 3>& grad_theta, State& state,
                                           const tensor<double, 3, 3>& grad_u_old, double theta_old, double dt)
  {
    auto [P, cv0, s0, q0] = calculateConstitutiveOutputs(grad_u, theta, grad_theta, state, grad_u_old, theta_old, dt);
    return serac::tuple{cv0, s0, q0};
  }

  /**
   * @brief Return constitutive output for mechanics operator
   *
   * @param[in] grad_u displacement gradient
   * @param[in] theta temperature
   * @param[in] grad_theta temperature gradient
   * @param[in] grad_u_old displacement gradient at previous time step
   * @param[in] theta_old temperature at previous time step
   * @param[in] dt time increment
   * @param[out] P Piola stress
   */
  auto calculateMechanicalConstitutiveOutputs(const tensor<double, 3, 3>& grad_u, double theta,
                                              const tensor<double, 3>& grad_theta, State& state,
                                              const tensor<double, 3, 3>& grad_u_old, double theta_old, double dt)
  {
    auto [P, cv0, s0, q0] = calculateConstitutiveOutputs(grad_u, theta, grad_theta, state, grad_u_old, theta_old, dt);
    return P;
  }

  /**
   * @brief evaluate free energy density
   * @param[in] grad_u displacement gradient
   * @param[in] theta temperature
   */
  template <typename T1, typename T2>
  auto calculateFreeEnergy(const tensor<T1, 3, 3>& grad_u, T2 theta)
  {
    const double K      = E / (3.0 * (1.0 - 2.0 * nu));
    const double G      = 0.5 * E / (1.0 + nu);
    auto         strain = greenStrain(grad_u);
    auto         trE    = tr(strain);
    auto         psi_1  = G * squared_norm(dev(strain)) + 0.5 * K * trE * trE;
    using std::log;
    auto logT  = log(theta / theta_ref);
    auto psi_2 = C * (theta - theta_ref + theta * logT);
    auto psi_3 = -3.0 * K * alpha * (theta - theta_ref) * trE;
    return psi_1 + psi_2 + psi_3;
  }
};

TEST(LiqCrystElastMaterial, FreeEnergyIsZeroInReferenceState)
{
  LCEMaterialProperties material{.E = 100.0, .nu = 0.25, .C = 1.0, .alpha = 1.0e-3, .theta_ref = 300.0, .k = 1.0};
  tensor<double, 3, 3>  displacement_grad{};
  double                temperature = material.theta_ref;
  double                free_energy = material.calculateFreeEnergy(displacement_grad, temperature);
  EXPECT_NEAR(free_energy, 0.0, 1e-10);
}

TEST(LiqCrystElastMaterial, StressIsZeroInReferenceState)
{
  LCEMaterialProperties material{.E = 100.0, .nu = 0.25, .C = 1.0, .alpha = 1.0e-3, .theta_ref = 300.0, .k = 1.0};
  tensor<double, 3, 3>  displacement_grad{};
  double                temperature = material.theta_ref;
  tensor<double, 3>     temperature_grad{};
  LCEMaterialProperties::State state{};
  auto                         displacement_grad_old = displacement_grad;
  double                       temperature_old       = temperature;
  double                       dt                    = 1.0;
  auto stress = material.calculateMechanicalConstitutiveOutputs(displacement_grad, temperature, temperature_grad, state,
                                                                displacement_grad_old, temperature_old, dt);
  EXPECT_NEAR(norm(stress), 0.0, 1e-10);
}

TEST(LiqCrystElastMaterial, FreeEnergyAndStressAgree)
{
  LCEMaterialProperties material{.E = 100.0, .nu = 0.25, .C = 1.0, .alpha = 1.0e-3, .theta_ref = 300.0, .k = 1.0};
  // clang-format off
  tensor<double, 3, 3>  displacement_grad{{{0.35490513, 0.60419905, 0.4275843},
                                           {0.23061597, 0.6735498,  0.43953657},
                                           {0.25099766, 0.27730572, 0.7678207}}};
  // clang-format on
  double                       temperature = 290.0;
  tensor<double, 3>            temperature_grad{0.87241435, 0.11105156, -0.27708054};
  LCEMaterialProperties::State state{};
  tensor<double, 3, 3>         displacement_grad_old{};
  double                       temperature_old = temperature;
  double                       dt              = 1.0;
  auto energy_and_stress = material.calculateFreeEnergy(make_dual(displacement_grad), temperature);
  auto stress = material.calculateMechanicalConstitutiveOutputs(displacement_grad, temperature, temperature_grad, state,
                                                                displacement_grad_old, temperature_old, dt);
  auto error  = stress - get_gradient(energy_and_stress);
  EXPECT_NEAR(norm(error), 0.0, 1e-12);
}

}  // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();

  return result;
}

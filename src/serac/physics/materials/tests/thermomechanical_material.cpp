#include <iostream>

#include <gtest/gtest.h>

#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/tuple.hpp"
#include "serac/numerics/functional/tuple_arithmetic.hpp"


namespace serac {

static constexpr auto I = Identity<3>();

static const tensor<double, 3, 3> Q{{{ 0.852932108456964,  0.405577416697288, -0.328654495524269 },
                                     {-0.51876824611259 ,  0.728725283562492, -0.44703351990878  },
                                     { 0.058192140263313,  0.551784758906877,  0.831953877717782 }}};

template <typename T>
auto green_strain(const T& grad_u)
{
  return 0.5*(grad_u + transpose(grad_u) + transpose(grad_u)*grad_u);
}


// maybe the diag operator would be nice

/**
 * Very incorrectly computes the matrix logarithm.
 * This only works for diagonal matrices, but
 * that is all I need for my test right now.
 * We'll need the real implementation in the
 * tensor class eventually, so beg Sam.
 */
template <typename T>
auto logm(const T& A)
{
  T logA{{{std::log(A[0][0]), 0.0, 0.0},
          {0.0, std::log(A[1][1]), 0.0},
          {0.0, 0.0, std::log(A[2][2])}}};
  return logA;
}

template <typename T>
auto hughes_winget_rate(const T& A, double dt)
{
  return (2./dt)*(A - I)*inv(A + I);
}

template <typename T>
auto compute_velocity_gradient(const T& F_new, const T& F_old, double dt)
{
  auto dF = F_new*inv(F_old); // this could be done with linear solves to avoid the inverse
  return hughes_winget_rate(dF, dt);
}

struct LinearThermoelasticMaterial {
  double E;      ///< Young's modulus
  double nu;     ///< Poisson's ratio
  double C;      ///< heat capacity
  double alpha;  ///< thermal expansion coefficient
  double theta_ref;   ///< datum temperature
  double k;      ///< thermal conductivity

  auto calculate_potential(const tensor<double, 3, 3>& grad_u, double theta,
                           const tensor<double, 3>& /* grad_theta */)
  {
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);
    auto strain = green_strain(grad_u);
    auto trE = tr(strain);
    auto psi_e = G*sqnorm(dev(strain)) + 0.5*K*trE*trE;
    using std::log;
    auto logT = log(theta/theta_ref);
    auto psi_t = C*(theta - theta_ref + theta*logT);
    auto psi_inter = -3.0*K*alpha*(theta - theta_ref)*trE;
    return psi_e + psi_t + psi_inter;
  }

  auto calculate_stress(const tensor<double, 3, 3>& grad_u, double theta,
                        const tensor<double, 3>& /* grad_theta */)
  {
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);
    auto Eg = green_strain(grad_u);
    auto trEg = tr(Eg);
    auto S = 2.0*G*dev(Eg) + K*(trEg - 3.0*alpha*(theta - theta_ref))*I;
    auto F = grad_u + I;
    return F*S;
  }

  template <typename T1, typename T2, typename T3>
  auto calculate_potential_and_stress_AD(const T1& grad_u, T2 theta,
                                         const T3& /* grad_theta */)
  {
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);
    auto strain = green_strain(grad_u);
    auto trE = tr(strain);
    auto psi_e = G*sqnorm(dev(strain)) + 0.5*K*trE*trE;
    using std::log;
    auto logT = log(theta/theta_ref);
    auto psi_t = C*(theta - theta_ref + theta*logT);
    auto psi_inter = -3.0*K*alpha*(theta - theta_ref)*trE;
    return psi_e + psi_t + psi_inter;
  }


  auto calculate_thermal_constitutive(const tensor<double, 3, 3>& grad_u,
                                      double theta,
                                      const tensor<double, 3>& grad_theta,
                                      const tensor<double, 3, 3>& grad_u_old,
                                      double /*theta_old*/,
                                      double dt)
  {
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    auto F = grad_u + I;
    auto F_old = grad_u_old + I;
    auto L = compute_velocity_gradient(F, F_old, dt);
    auto D = sym(L);
    auto strain_rate = transpose(F_old)*D*F_old;
    double src = -3*K*alpha*theta*tr(strain_rate);
    auto q = -k*grad_theta;
    return serac::tuple{C, src, q};
  }
  
  auto calculate_constitutive_output(const tensor<double, 3, 3>& grad_u,
                                     double theta,
                                     const tensor<double, 3>& grad_theta,
                                     const tensor<double, 3, 3>& grad_u_old,
                                     double /*theta_old*/,
                                     double dt)
  {
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);
    const auto Eg = green_strain(grad_u);
    const auto trEg = tr(Eg);
    const auto F = grad_u + I;
    const auto F_old = grad_u_old + I;
    const auto D = sym(compute_velocity_gradient(F, F_old, dt));
    const auto Eg_dot = transpose(F_old)*D*F_old;

    // stress
    const auto S = 2.0*G*dev(Eg) + K*(trEg - 3.0*alpha*(theta - theta_ref))*I;
    const auto P = F*S;

    // internal heat source
    const double src = -3*K*alpha*theta*tr(Eg_dot);

    // heat flux
    const auto q = -k*grad_theta;

    return serac::tuple{P, C, src, q};
  }

  auto calculate_thermal_constitutive_outputs(
      const tensor<double, 3, 3>& grad_u, double theta, const tensor<double, 3>& grad_theta,
      const tensor<double, 3, 3>& grad_u_old, double theta_old, double dt)
  {
    auto [P, c, s, q] = calculate_constitutive_output(grad_u, theta, grad_theta, grad_u_old,
                                                      theta_old, dt);
    return serac::tuple{c, s, q};
  }

  auto calculate_mechanical_constitutive_outputs(
      const tensor<double, 3, 3>& grad_u, double theta, const tensor<double, 3>& grad_theta,
      const tensor<double, 3, 3>& grad_u_old, double theta_old, double dt)
  {
    auto [P, c, s, q] = calculate_constitutive_output(grad_u, theta, grad_theta, grad_u_old,
                                                      theta_old, dt);
    return P;
    
  }
};

  

TEST(ThermomechanicalMaterial, FreeEnergyIsZeroInReferenceState)
{
  LinearThermoelasticMaterial material{.E=100.0, .nu=0.25, .C=1.0, .alpha=1.0e-3, .theta_ref=300.0, .k=1.0};
  tensor<double, 3, 3> displacement_grad;
  double temperature = material.theta_ref;
  tensor<double, 3> temperature_grad{0.0, 0.0, 0.0};
  double free_energy = material.calculate_potential(displacement_grad, temperature, temperature_grad);
  EXPECT_NEAR(free_energy, 0.0, 1e-10);
}

TEST(ThermomechanicalMaterial, StressIsZeroInReferenceState)
{
  LinearThermoelasticMaterial material{.E=100.0, .nu=0.25, .C=1.0, .alpha=1.0e-3, .theta_ref=300.0, .k=1.0};
  tensor<double, 3, 3> displacement_grad = serac::zero{};
  double temperature = material.theta_ref;
  tensor<double, 3> temperature_grad = serac::zero{};
  auto displacement_grad_old = displacement_grad;
  double temperature_old = temperature;
  double dt = 1.0;
  auto stress = material.calculate_mechanical_constitutive_outputs(
      displacement_grad, temperature, temperature_grad, displacement_grad_old, temperature_old, dt);
  EXPECT_NEAR(norm(stress), 0.0, 1e-10);
}

TEST(ThermomechanicalMaterial, MechanicalConstitutiveOutputsAreFrameInvariant)
{
  LinearThermoelasticMaterial material{.E=100.0, .nu=0.25, .C=1.0, .alpha=1.0e-3, .theta_ref=300.0, .k=1.0};
  tensor<double, 3, 3> displacement_grad{{{0.35490513, 0.60419905, 0.4275843 },
                                          {0.23061597, 0.6735498 , 0.43953657},
                                          {0.25099766, 0.27730572, 0.7678207 }}};
  double temperature = material.theta_ref;
  tensor<double, 3> temperature_grad{0.87241435, 0.11105156, -0.27708054};
  auto energy = material.calculate_potential(displacement_grad, temperature, temperature_grad);
  auto displacement_grad_transformed = Q*(displacement_grad + I) - I;
  auto temperature_grad_transformed = Q*temperature_grad;
  double energy2 = material.calculate_potential(displacement_grad_transformed,
                                                temperature,
                                                temperature_grad_transformed);
  EXPECT_NEAR(energy, energy2, 1e-12);
}


TEST(ThermomechanicalMaterial, EnergyAndStressAgree)
{
  LinearThermoelasticMaterial material{.E=100.0, .nu=0.25, .C=1.0, .alpha=1.0e-3, .theta_ref=300.0, .k=1.0};
  tensor<double, 3, 3> displacement_grad{{{0.35490513, 0.60419905, 0.4275843 },
                                          {0.23061597, 0.6735498 , 0.43953657},
                                          {0.25099766, 0.27730572, 0.7678207 }}};
  double temperature = 290.0;
  tensor<double, 3> temperature_grad{0.87241435, 0.11105156, -0.27708054};
  tensor<double, 3, 3> displacement_grad_old = serac::zero{};
  double temperature_old = temperature;
  double dt = 1.0;
  auto energy_and_stress = material.calculate_potential_and_stress_AD(make_dual(displacement_grad), temperature, temperature_grad);
  auto stress = material.calculate_mechanical_constitutive_outputs(
      displacement_grad, temperature, temperature_grad, displacement_grad_old,
      temperature_old, dt);
  auto error = stress - get_gradient(energy_and_stress);
  EXPECT_NEAR(norm(error), 0.0, 1e-12);
}

TEST(ThermomechanicalMaterial, SatisfiesDissipationInequality)
{
  LinearThermoelasticMaterial material{.E=100.0, .nu=0.25, .C=1.0, .alpha=1.0e-3, .theta_ref=300.0, .k=1.0};
  tensor<double, 3, 3> displacement_grad{{{0.35490513, 0.60419905, 0.4275843 },
                                          {0.23061597, 0.6735498 , 0.43953657},
                                          {0.25099766, 0.27730572, 0.7678207 }}};
  double temperature = 290.0;
  tensor<double, 3> temperature_grad{0.87241435, 0.11105156, -0.27708054};
  double temperature_old = temperature;
  tensor<double, 3, 3> displacement_grad_old = serac::zero{};
  double dt = 1.0;
  auto generalized_fluxes = material.calculate_thermal_constitutive_outputs(
      displacement_grad, temperature, temperature_grad, displacement_grad_old,
      temperature_old, dt);
  auto [heat_capacity, source, heat_flux] = generalized_fluxes;
  // "inner" didn't work for me, but "dot" does
  // ask sam and jamie about this
  auto dissipation = -dot(heat_flux, temperature_grad)/temperature;
  EXPECT_TRUE(dissipation >= 0.0);
}

TEST(ThermomechanicalMaterial, ThermalConstitutiveOutputsAreFrameInvariant)
{
  LinearThermoelasticMaterial material{.E=100.0, .nu=0.25, .C=1.0, .alpha=1.0e-3, .theta_ref=300.0, .k=1.0};
  tensor<double, 3, 3> displacement_grad{{{0.35490513, 0.60419905, 0.4275843 },
                                          {0.23061597, 0.6735498 , 0.43953657},
                                          {0.25099766, 0.27730572, 0.7678207 }}};
  double temperature = 290.0;
  tensor<double, 3> temperature_grad{0.87241435, 0.11105156, -0.27708054};
  double temperature_old = temperature;
  tensor<double, 3, 3> displacement_grad_old = serac::zero{};
  double dt = 1.0;
  auto displacement_grad_transformed = Q*(displacement_grad + I) - I;
  auto displacement_grad_old_transformed = Q*(displacement_grad_old + I) - I;
  auto generalized_fluxes = material.calculate_thermal_constitutive_outputs(
      displacement_grad, temperature, temperature_grad, displacement_grad_old,
      temperature_old, dt);
  auto generalized_fluxes_2 = material.calculate_thermal_constitutive_outputs(
      displacement_grad_transformed, temperature, temperature_grad,
      displacement_grad_old_transformed, temperature_old, dt);
  auto [heat_capacity, internal_source, heat_flux] = generalized_fluxes;
  auto [heat_capacity_2, internal_source_2, heat_flux_2] = generalized_fluxes_2;
  EXPECT_NEAR(heat_capacity, heat_capacity_2, 1e-12);
  EXPECT_NEAR(internal_source, internal_source_2, 1e-12);
  EXPECT_LT(norm(heat_flux - heat_flux_2), 1e-12); 
}


TEST(ThermomechanicalMaterial, InternalSourceHasCorrectSign)
{
  LinearThermoelasticMaterial material{.E=100.0, .nu=0.25, .C=1.0, .alpha=1.0e-3, .theta_ref=300.0, .k=1.0};
  tensor<double, 3, 3> displacement_grad{{{0.35490513, 0.60419905, 0.4275843 },
                                          {0.23061597, 0.6735498 , 0.43953657},
                                          {0.25099766, 0.27730572, 0.7678207 }}};
  double temperature_old = 290.0;
  tensor<double, 3> temperature_grad{0.87241435, 0.11105156, -0.27708054};
  tensor<double, 3, 3> displacement_grad_old = serac::zero{};
  double temperature = temperature_old;
  double dt = 1.0;
  auto [heat_capacity, internal_source, heat_flux] =
      material.calculate_thermal_constitutive_outputs(
          displacement_grad, temperature, temperature_grad, displacement_grad_old,
          temperature_old, dt);
  EXPECT_LT(internal_source, 0.0);
}


} // namespace serac


int main (int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);

  int result = RUN_ALL_TESTS();

  return result;
}

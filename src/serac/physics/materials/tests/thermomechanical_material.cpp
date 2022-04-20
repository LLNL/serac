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
  
};

  

TEST(thermomechanical_material, energyZeroPoint)
{
  LinearThermoelasticMaterial material{.E=100.0, .nu=0.25, .C=1.0, .alpha=1.0e-3, .theta_ref=300.0, .k=1.0};
  tensor<double, 3, 3> displacement_grad;
  double temperature = 300.0;
  tensor<double, 3> temperature_grad{0.0, 0.0, 0.0};
  double free_energy = material.calculate_potential(displacement_grad, temperature, temperature_grad);
  EXPECT_NEAR(free_energy, 0.0, 1e-10);
}

TEST(thermomechanical_material, stressZeroPoint)
{
  LinearThermoelasticMaterial material{.E=100.0, .nu=0.25, .C=1.0, .alpha=1.0e-3, .theta_ref=300.0, .k=1.0};
  tensor<double, 3, 3> displacement_grad;
  double temperature = 300.0;
  tensor<double, 3> temperature_grad{0.0, 0.0, 0.0};
  auto stress = material.calculate_stress(displacement_grad, temperature, temperature_grad);
  EXPECT_NEAR(norm(stress), 0.0, 1e-10);
}

TEST(thermomechanical_material, frameInvariance)
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


TEST(thermomechanical_material, energyStressAgreement)
{
  LinearThermoelasticMaterial material{.E=100.0, .nu=0.25, .C=1.0, .alpha=1.0e-3, .theta_ref=300.0, .k=1.0};
  tensor<double, 3, 3> displacement_grad{{{0.35490513, 0.60419905, 0.4275843 },
                                          {0.23061597, 0.6735498 , 0.43953657},
                                          {0.25099766, 0.27730572, 0.7678207 }}};
  double temperature = 290.0;
  tensor<double, 3> temperature_grad{0.87241435, 0.11105156, -0.27708054};
  auto energy_and_stress = material.calculate_potential_and_stress_AD(make_dual(displacement_grad), temperature, temperature_grad);
  auto stress = material.calculate_stress(displacement_grad, temperature, temperature_grad);
  auto error = stress - get_gradient(energy_and_stress);
  std::cout << "P1 = " << stress <<"\n";
  std::cout << "P2 = " << get_gradient(energy_and_stress) << std::endl;
  EXPECT_NEAR(norm(error), 0.0, 1e-12);
}

}


int main (int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);

  int result = RUN_ALL_TESTS();

  return result;
}

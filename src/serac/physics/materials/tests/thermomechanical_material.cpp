#include <iostream>

#include <gtest/gtest.h>

#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/tuple.hpp"
#include "serac/numerics/functional/tuple_arithmetic.hpp"


namespace serac {

auto green_strain(const tensor<double, 3, 3>& grad_u)
{
  return 0.5*(grad_u + transpose(grad_u) + dot(grad_u, transpose(grad_u)));
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
    auto psi_e = G*sqnorm(strain) + 0.5*K*trE*trE;
    using std::log;
    auto logT = log(theta/theta_ref);
    auto psi_t = C*(theta - theta_ref + theta*logT);
    auto psi_inter = -3.0*K*alpha*(theta - theta_ref)*trE;
    return psi_e + psi_t + psi_inter;
  }
  
};

  

TEST(thermomechanical_material, zeroPoint) {
  LinearThermoelasticMaterial material{.E=100.0, .nu=0.25, .C=1.0, .alpha=1.0e-3, .theta_ref=300.0, .k=1.0};
  tensor<double, 3, 3> displacement_grad;
  double temperature = 300.0;
  tensor<double, 3> temperature_grad{0.0, 0.0, 0.0};
  double free_energy = material.calculate_potential(displacement_grad, temperature, temperature_grad);
  EXPECT_NEAR(free_energy, 0.0, 1e-10);
}

}


int main (int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);

  int result = RUN_ALL_TESTS();

  return result;
}

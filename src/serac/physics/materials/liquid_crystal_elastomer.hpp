// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file liquid_crystal_elastomer.hpp
 *
 * @brief Brighenti's constitutive model for liquid crystal elastomers
 *
 * see https://doi.org/10.1016/j.ijsolstr.2021.02.023
 */

#pragma once

#include "serac/numerics/functional/tuple.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/dual.hpp"
#include "serac/physics/materials/solid_functional_material.hpp"

namespace serac {

/**
 * @brief Brighenti liquid crystal elastomer model
 *
 *
 */
class LiquidCrystalElastomer {
public:

  static constexpr int dim = 3;

  struct State {
    tensor<double, dim, dim> deformation_gradient;
    tensor<double, dim, dim> distribution_tensor;
    double temperature;
  };
  
  /**
   * @brief Constructor
   *
   * @param density Density of the material (in reference configuration)
   * @param shear_modulus Shear modulus of the material
   * @param bulk_modulus Bulk modulus of the material
   * @param order_constant temperature-valued constant in exponential factor for order parameter
   * @param order_parameter Initial value of the order parameter
   * @param transition_temperature Characteristic temperature of the order-disorder transition
   * @param normal Liquid crystal director vector
   * @param N_b_squared Number of Kunh segments/chain, times square of Kuhn segment length
   */
  LiquidCrystalElastomer(double density, double shear_modulus, double bulk_modulus,
                         double order_constant, double order_parameter,
                         double transition_temperature, tensor<double, 3> normal,
                         double N_b_squared):
    density_(density),
    shear_modulus_(shear_modulus),
    bulk_modulus_(bulk_modulus),
    order_constant_(order_constant),
    initial_order_parameter_(order_parameter),
    transition_temperature_(transition_temperature),
    normal_(normal/norm(normal)),
    N_b_squared_(N_b_squared),
    initial_distribution_tensor_(
        calculateInitialDistributionTensor(normal, order_parameter, N_b_squared))
  {
    SLIC_ERROR_ROOT_IF(density_ <= 0.0, "Density must be positive in the LCE material model.");
    SLIC_ERROR_ROOT_IF(shear_modulus_ <= 0.0, "Shear modulus must be positive in the LCE material model.");
    SLIC_ERROR_ROOT_IF(bulk_modulus_ <= 0.0, "Bulk modulus must be positive in the LCE material model.");
    SLIC_ERROR_ROOT_IF(order_constant_ <= 0.0, "Order constant must be positive in the LCE material model.");
    SLIC_ERROR_ROOT_IF(transition_temperature_ <= 0.0, "The transition temperature must be positive in the LCE material model.");
  }

  /**
   * @brief Material response
   *
   * @tparam DisplacementType number-like type for the displacement vector
   * @tparam DispGradType number-like type for the displacement gradient tensor
   *
   * @param displacement_grad displacement gradient with respect to the reference configuration
   * @param[in,out] state A state variable object for this material. The value is updated in place.
   * @param temperature the temperature
   * @return The calculated material response (density, Kirchoff stress) for the material
   */
  template <typename DisplacementType, typename DispGradType>
  SERAC_HOST_DEVICE auto operator()(
      const tensor<double, dim>& /* x */,  const tensor<DisplacementType, dim>& /*displacement*/,
      const tensor<DispGradType, dim, dim>& displacement_grad, State& state, double temperature) const
  {
    // kinematics
    auto I     = Identity<dim>();
    auto F     = displacement_grad + I;
    auto F_old = state.deformation_gradient;
    auto F_hat = dot(F, inv(F_old));
    auto J = det(F);

    // Distribution tensor function of nematic order tensor
    auto mu = calculateDistributionTensor(F_hat, temperature, state);
    auto& mu0 = initial_distribution_tensor_;

    // stress output
    // Note to Jorge-Luis: the paper is omits a prefactor of 1/J in the
    // Cauchy stress equation because they assume J = 1 strictly
    // (the incompressible limit). It needs to be retained for this
    // compressible model (and then cancelled out when converting to
    // Kirchhoff stress).
    auto stress = (3*shear_modulus_/N_b_squared_) * (mu - mu0);
    const double lambda = bulk_modulus_ - 2.0 / 3.0 * shear_modulus_;
    using std::log;
    stress = stress + lambda*log(J)*I;

    // update state variables
    state.deformation_gradient = get_value(F);
    state.temperature = temperature;
    state.distribution_tensor = get_value(mu);
    
    return solid_util::MaterialResponse{density_, stress};
  }

/// -------------------------------------------------------

  static tensor<double, dim, dim>
  calculateInitialDistributionTensor(const tensor<double, dim> normal, double q0, double N_b_squared)
  {
    // Initial distribution tensor
    auto I      = DenseIdentity<dim>();
    auto mu_0_a = (1 - q0) * I;
    auto mu_0_b = 3 * q0 * outer(normal, normal);

    return N_b_squared/3 * (mu_0_a + mu_0_b);
  }

/// -------------------------------------------------------

  template <typename T>
  auto calculateDistributionTensor(
    const tensor<T, dim, dim> F_hat, const double theta, const State& state) const
  {
    // Nematic order scalar
    auto theta_old = state.temperature;
    double q_old = initial_order_parameter_ / (1 + std::exp((theta_old - transition_temperature_)/order_constant_));
    double q     = initial_order_parameter_ / (1 + std::exp((theta - transition_temperature_)/order_constant_));

    // Nematic order tensor
    auto proj = outer(normal_, normal_) - Identity<dim>();
    auto Q_old = q_old/2 * (3 * proj);
    auto Q     = q/2 * (3 * proj);

    // Polar decomposition of incremental deformation gradient
    auto U_hat = tensorSquareRoot(transpose(F_hat) * F_hat);
    auto R_hat = F_hat * inv(U_hat);
    
    // Distribution tensor (using 'Strang Splitting' approach)
    double alpha = 2.*N_b_squared_/3.;
    auto mu_old = state.distribution_tensor;
    auto mu_hat = mu_old + alpha * (Q - Q_old);
    auto mu_a = dot(F_hat, dot(mu_hat, transpose(F_hat)));
    auto mu_b = alpha * (Q - dot(R_hat, dot(Q, transpose(R_hat))));
    
    return mu_a + mu_b;
  }

/// -------------------------------------------------------

  template <typename T>
  auto tensorSquareRoot(const tensor<T, dim, dim>& A) const
  {
    auto X = A;
    for (int i = 0; i < 15; i++) {
      X = 0.5 * (X + dot(A, inv(X)));
    }
    return X;
  }

private:
  // Density
  double density_;

  // elastic moduli in the stress free configuration
  double shear_modulus_;
  double bulk_modulus_;
  
  // Order constant
  double order_constant_;

  // intial value of order parameter
  double initial_order_parameter_;

  // Transition temperature
  double transition_temperature_;

  // director vector of the liquid crystal orientation
  tensor<double, 3> normal_;

  // Kuhn segment parameters.
  // BT: I think this can be removed - it looks like it cancels out every place it appears.
  double N_b_squared_;

  // initial value of the distribution tensor
  tensor<double, 3, 3> initial_distribution_tensor_;
};

}  // namespace serac

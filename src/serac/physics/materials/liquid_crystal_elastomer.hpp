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
#include "serac/physics/materials/solid_material.hpp"

namespace serac {

/**
 * @brief Brighenti liquid crystal elastomer model
 */
struct LiquidCrystalElastomer {
  /// this model is only intended to be used in 3D
  static constexpr int dim = 3;

  /// internal variables for the liquid crystal elastomer model
  struct State {
    tensor<double, dim, dim> deformation_gradient;  ///< F from the last timestep
    tensor<double, dim, dim> distribution_tensor;   ///< mu from the last timestep
    double                   temperature;           ///< temperature at the last timestep
  };

  /**
   * @brief Constructor
   *
   * @param rho mass density of the material (in reference configuration)
   * @param shear_modulus Shear modulus of the material
   * @param bulk_modulus Bulk modulus of the material
   * @param order_constant temperature-valued constant in exponential factor for order parameter
   * @param order_parameter Initial value of the order parameter
   * @param transition_temperature Characteristic temperature of the order-disorder transition
   * @param N_b_squared Number of Kunh segments/chain, times square of Kuhn segment length
   */
  LiquidCrystalElastomer(double rho, double shear_modulus, double bulk_modulus, double order_constant,
                         double order_parameter, double transition_temperature, double N_b_squared)
      : density(rho),
        shear_modulus_(shear_modulus),
        bulk_modulus_(bulk_modulus),
        order_constant_(order_constant),
        initial_order_parameter_(order_parameter),
        transition_temperature_(transition_temperature),
        N_b_squared_(N_b_squared)
  {
    SLIC_ERROR_ROOT_IF(density <= 0.0, "Density must be positive in the LCE material model.");
    SLIC_ERROR_ROOT_IF(shear_modulus_ <= 0.0, "Shear modulus must be positive in the LCE material model.");
    SLIC_ERROR_ROOT_IF(bulk_modulus_ <= 0.0, "Bulk modulus must be positive in the LCE material model.");
    SLIC_ERROR_ROOT_IF(order_constant_ <= 0.0, "Order constant must be positive in the LCE material model.");
    SLIC_ERROR_ROOT_IF(transition_temperature_ <= 0.0,
                       "The transition temperature must be positive in the LCE material model.");
  }

  /**
   * @brief Material response
   *
   * @tparam DisplacementType number-like type for the displacement vector
   * @tparam DispGradType number-like type for the displacement gradient tensor
   * @tparam AngleType number-like type for the orientation angle gamma
   *
   * @param[in,out] state A state variable object for this material. The value is updated in place.
   * @param[in] displacement_grad displacement gradient with respect to the reference configuration
   * @param[in] temperature the temperature
   * @param[in] gamma the polar angle used to define the liquid crystal orientation vector
   * @return The calculated material response (Cauchy stress) for the material
   */
  template <typename DispGradType, typename TemperatureType, typename AngleType>
  SERAC_HOST_DEVICE auto operator()(State& state, const tensor<DispGradType, dim, dim>& displacement_grad,
                                    TemperatureType temperature, AngleType gamma) const
  {
    using std::cos, std::sin;

    auto   I  = Identity<dim>();
    double q0 = initial_order_parameter_;
    tensor normal{{cos(gamma), sin(gamma), 0.0 * gamma}};

    if (norm(state.deformation_gradient) == 0) {
      state.distribution_tensor  = get_value((N_b_squared_ / 3.0) * ((1 - q0) * I) + (3 * q0 * outer(normal, normal)));
      state.deformation_gradient = get_value(displacement_grad) + I;
      state.temperature          = get_value(temperature);
    }

    // kinematics
    auto F     = displacement_grad + I;
    auto F_old = state.deformation_gradient;
    auto F_hat = dot(F, inv(F_old));
    auto J     = det(F);

    // Distribution tensor function of nematic order tensor
    auto mu  = calculateDistributionTensor(state, F_hat, temperature, normal);
    auto mu0 = (N_b_squared_ / 3.0) * ((1 - q0) * I) + (3 * q0 * outer(normal, normal));

    // stress output
    // Note to Jorge-Luis: the paper is omits a prefactor of 1/J in the
    // Cauchy stress equation because they assume J = 1 strictly
    // (the incompressible limit). It needs to be retained for this
    // compressible model.
    auto         stress = (3 * shear_modulus_ / N_b_squared_) * (mu - mu0);
    const double lambda = bulk_modulus_ - (2.0 / 3.0) * shear_modulus_;
    using std::log;
    stress += lambda * log(J) * I;
    stress /= det(J);

    // update state variables
    state.deformation_gradient = get_value(F);
    state.temperature          = get_value(temperature);
    state.distribution_tensor  = get_value(mu);

    return stress;
  }

  /// -------------------------------------------------------

  template <typename S, typename T, typename U>
  auto calculateDistributionTensor(const State& state, const tensor<S, dim, dim>& F_hat, const T theta,
                                   const tensor<U, dim>& normal) const
  {
    // Nematic order scalar
    using std::exp;
    auto theta_old = state.temperature;
    auto q_old     = initial_order_parameter_ / (1 + exp((theta_old - transition_temperature_) / order_constant_));
    auto q         = initial_order_parameter_ / (1 + exp((theta - transition_temperature_) / order_constant_));

    // Nematic order tensor
    constexpr auto I      = Identity<dim>();
    auto           n_dyad = outer(normal, normal);
    // BT: These are different than what Jorge-Luis had. I found the papers
    // to be confusing on this point. I'm extrapolating from Eq (7)
    // in https://doi.org/10.1016/j.mechrescom.2022.103858
    // Well-defined validation problems would help to confirm.
    auto Q_old = 0.5 * ((1.0 - q_old) * I + 3.0 * q_old * n_dyad);
    auto Q     = 0.5 * ((1.0 - q) * I + 3.0 * q * n_dyad);

    // Polar decomposition of incremental deformation gradient
    auto U_hat = matrix_sqrt(transpose(F_hat) * F_hat);
    auto R_hat = F_hat * inv(U_hat);

    // Distribution tensor (using 'Strang Splitting' approach)
    double alpha  = 2.0 * N_b_squared_ / 3.0;
    auto   mu_old = state.distribution_tensor;
    auto   mu_hat = mu_old + alpha * (Q - Q_old);
    auto   mu_a   = dot(F_hat, dot(mu_hat, transpose(F_hat)));
    auto   mu_b   = alpha * (Q - dot(R_hat, dot(Q, transpose(R_hat))));

    return mu_a + mu_b;
  }

  /// -------------------------------------------------------

  // Sam: please forgive some of the tautological
  // explanations below, I'm not knowledgeable enough
  // about this model to write meaningful descriptions,
  // so these placeholders really only exist to satisfy
  // our doxygen requirements
  //
  // suggestions are welcome

  double density;                   ///<  mass density
  double shear_modulus_;            ///< shear modulus in stress-free configuration
  double bulk_modulus_;             ///< bulk modulus in stress-free configuration
  double order_constant_;           ///< Order constant
  double initial_order_parameter_;  ///< initial value of order parameter
  double transition_temperature_;   ///< Transition temperature

  // BT: I think this can be removed - it looks like it cancels out every place it appears.
  double N_b_squared_;  ///< Kuhn segment parameters.
};

}  // namespace serac

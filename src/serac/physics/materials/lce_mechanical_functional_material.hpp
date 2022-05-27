// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_functional_material.hpp
 *
 * @brief The material and load types for the solid functional physics module
 */

#pragma once

#include "serac/numerics/functional/functional.hpp"

/// lce_mechanical_util helper data types
namespace serac::lce_mechanical_util{

/**
 * @brief Response data type for solid mechanics simulations
 *
 * @tparam DensityType Density type
 * @tparam StressType Stress type (i.e. second order tensor)
 */
template <typename DensityType, typename StressType>
struct MaterialResponse {
  /// Density of the material (mass/volume)
  DensityType density;

  /// Kirchoff stress (det(deformation gradient) * Cauchy stress) for the constitutive model
  StressType stress;
};

/**
 * @brief Template deduction guide for the material response
 *
 * @tparam DensityType Density type
 * @tparam StressType Stress type (i.e. second order tensor)
 */
template <typename DensityType, typename StressType>
MaterialResponse(DensityType, StressType) -> MaterialResponse<DensityType, StressType>;

/// -------------------------------------------------------

/**
 * @brief Neo-Hookean material model
 *
 * @tparam dim The spatial dimension of the mesh
 */
template <int dim>
class BrighentiMechanical {
public:
  /**
   * @brief Construct a new Neo-Hookean object
   *
   * @param density Density of the material
   * @param shear_modulus Shear modulus of the material
   * @param bulk_modulus Bulk modulus of the material
   */
  BrighentiMechanical(
    double density = 1.0, 
    double shear_modulus = 1.0, 
    double order_constant = 1.0, 
    double order_parameter = 1.0,
    double transition_temperature = 1.0,
    double hydrostatic_pressure = 1.0): 
    density_(density),
    shear_modulus_(shear_modulus),
    order_constant_(order_constant),
    order_parameter_(order_parameter),
    transition_temperature_(transition_temperature),
    hydrostatic_pressure_(hydrostatic_pressure),
    N_seg_(1.0),
    b_seg_(1.0)
  {
    SLIC_ERROR_ROOT_IF(density_ < 0.0, "Density must be positive in the LCE material model.");
    SLIC_ERROR_ROOT_IF(shear_modulus_ < 0.0, "Shear modulus must be positive in the LCE material model.");
    SLIC_ERROR_ROOT_IF(order_constant_ < -1.0, "Order constant must be greater than -1 in the LCE material model.");
    SLIC_ERROR_ROOT_IF(transition_temperature_ < 0.0, "The transition temperaturemust be positive in the LCE material model.");
  }

  /**
   * @brief Material response call for a neo-Hookean solid
   *
   * @tparam PositionType Spatial position type
   * @tparam DisplacementType Displacement type
   * @tparam DispGradType Displacement gradient type
   * @param displacement_grad displacement gradient with respect to the reference configuration (displacement_grad)
   * @return The calculated material response (density, Kirchoff stress) for the material
   */
  template <typename DisplacementType, typename DispGradType>
  SERAC_HOST_DEVICE auto operator()(
    const tensor<double, dim>& /* x */, 
    const DisplacementType& /*displacement*/,
    const DispGradType& displacement_grad) const
  {
    // Deformation gradients
    auto I     = Identity<dim>();
    auto F     = displacement_grad + I;
    auto F_old = displacement_grad*0.9 + I;
    auto F_hat = F * inv(F_old);

    // Determinant of deformation gradient
    [[maybe_unused]] auto J = det(F);

    // Distribution tensor function of nematic order tensor
    // tensor<double, 1, dim> normal = {2/std::sqrt(14), -1/std::sqrt(14), 3/std::sqrt(14)}};
    tensor<double, 1, dim> normal;
    normal[0][0] = 2/std::sqrt(14);
    normal[0][1] = -1/std::sqrt(14);

    if(dim>2)
    {
      normal[0][2] = 3/std::sqrt(14);
    }
    
    // auto I      = Identity<dim>();
    // auto mu_0_a = (1 - order_parameter_) * I;
    // auto mu_0_b = 3 * order_parameter_ * (transpose(normal) * normal);   
    // auto mu_0 = N_seg_*std::pow(b_seg_,2)/3 * (mu_0_a + mu_0_b); 

    // auto mu_0 = calculateInitialDistributionTensor(normal);

    double theta = 330;
    double theta_old = 328;
    auto mu = calculateDistributionTensor(normal, F_hat, theta, theta_old);

    // stress output
    auto stress = J * ( (3*shear_modulus_/(N_seg_*std::pow(b_seg_,2))) * (mu) + J*hydrostatic_pressure_*I ) * inv(transpose(F));
    // auto stress = J * ( (3*shear_modulus_/(N_seg_*std::pow(b_seg_,2))) * (mu_0) + J*hydrostatic_pressure_*I ) * inv(transpose(F_hat));

    return MaterialResponse{density_, stress};
  }

/// -------------------------------------------------------

  auto calculateInitialDistributionTensor(const tensor<double, 1, dim> normal) const
  {
    // Initial distribution tensor
    auto I      = Identity<dim>();
    auto mu_0_a = (1 - order_parameter_) * I;
    auto mu_0_b = 3 * order_parameter_ * (transpose(normal) * normal);

    return N_seg_*std::pow(b_seg_,2)/3 * (mu_0_a + mu_0_b);
  }

/// -------------------------------------------------------

  template <typename T1, typename T2>
  auto calculateDistributionTensor(
    tensor<double, 1, dim> normal,
    const tensor<T1, dim, dim> F_hat,
    const T2 theta,
    const double theta_old) const
  {
    auto I     = Identity<dim>();

    // Polar decomposition of deformation gradient based on F_hat
    auto U_hat = tensorSquareRoot(transpose(F_hat) * F_hat);
    auto R_hat = F_hat * inv(U_hat);

    // Nematic order scalar
    double q_old = order_parameter_ / (1 + std::exp((theta_old - transition_temperature_)/order_constant_));
    double q     = order_parameter_ / (1 + std::exp((theta - transition_temperature_)/order_constant_));

    // Nematic order tensor
    auto Q_old = q_old/2 * (3 * transpose(normal) * normal - I);
    auto Q     = q/2 * (3 * transpose(normal) * normal - I);
    
    // Distribution tensor (using 'Strang Splitting' approach)
    auto mu_old_a = (1 - q_old) * I;
    auto mu_old_b = 3 * q_old * (transpose(normal) * normal);
    auto mu_old   = N_seg_*std::pow(b_seg_,2)/3 * (mu_old_a + mu_old_b);

    auto mu_a = F_hat * ( mu_old + (2*N_seg_*std::pow(b_seg_,2)/3) * (Q - Q_old)) * transpose(F_hat);
    auto mu_b = (2*N_seg_*std::pow(b_seg_,2)/3) * (Q - R_hat * Q * transpose(R_hat));

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
  /// Density
  double density_;

  /// Shear modulus in the stress free configuration
  double shear_modulus_;

  /// Order constant
  double order_constant_;

  /// Order parameter
  double order_parameter_;

  /// Transition temperature
  double transition_temperature_;

  /// Hydrostatic pressure
  double hydrostatic_pressure_;

  double N_seg_;
  double b_seg_;
};

/// -------------------------------------------------------

/// Constant body force model
template <int dim>
struct ConstantBodyForce {
  /// The constant body force
  tensor<double, dim> force_;

  /**
   * @brief Evaluation function for the constant body force model
   *
   * @tparam DisplacementType Displacement type
   * @tparam DispGradType Displacement gradient type
   * @tparam dim The dimension of the problem
   * @return The body force value
   */
  template <typename DisplacementType, typename DispGradType>
  SERAC_HOST_DEVICE tensor<double, dim> operator()(const tensor<double, dim>& /* x */, const double /* t */,
                                                   const DisplacementType& /* displacement */,
                                                   const DispGradType& /* displacement_grad */) const
  {
    return force_;
  }
};

/// -------------------------------------------------------

/// Constant traction boundary condition model
template <int dim>
struct ConstantTraction {
  /// The constant traction
  tensor<double, dim> traction_;

  /**
   * @brief Evaluation function for the constant traction model
   *
   * @return The traction value
   */
  SERAC_HOST_DEVICE tensor<double, dim> operator()(const tensor<double, dim>& /* x */,
                                                   const tensor<double, dim>& /* n */, const double /* t */) const
  {
    return traction_;
  }
};

/// -------------------------------------------------------

/// Function-based traction boundary condition model
template <int dim>
struct TractionFunction {
  /// The traction function
  std::function<tensor<double, dim>(const tensor<double, dim>&, const tensor<double, dim>&, const double)>
      traction_func_;

  /**
   * @brief Evaluation for the function-based traction model
   *
   * @param x The spatial coordinate
   * @param n The normal vector
   * @param t The current time
   * @return The traction to apply
   */
  SERAC_HOST_DEVICE tensor<double, dim> operator()(const tensor<double, dim>& x, const tensor<double, dim>& n,
                                                   const double t) const
  {
    return traction_func_(x, n, t);
  }
};

/// -------------------------------------------------------

/// Constant pressure model
struct ConstantPressure {
  /// The constant pressure
  double pressure_;

  /**
   * @brief Evaluation of the constant pressure model
   *
   * @tparam dim Spatial dimension
   */
  template <int dim>
  SERAC_HOST_DEVICE double operator()(const tensor<double, dim>& /* x */, const double /* t */) const
  {
    return pressure_;
  }
};

/// -------------------------------------------------------

/// Function-based pressure boundary condition
template <int dim>
struct PressureFunction {
  /// The pressure function
  std::function<double(const tensor<double, dim>&, const double)> pressure_func_;

  /**
   * @brief Evaluation for the function-based pressure model
   *
   * @param x The spatial coordinate
   * @param t The current time
   * @return The pressure to apply
   */
  SERAC_HOST_DEVICE double operator()(const tensor<double, dim>& x, const double t) const
  {
    return pressure_func_(x, t);
  }
};

}  // namespace serac::lce_mechanical_util

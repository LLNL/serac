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

/// SolidFunctional helper data types
namespace serac::solid_util {

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

/**
 * @brief Linear isotropic elasticity material model
 *
 * @tparam dim Spatial dimension of the mesh
 */
template <int dim>
class LinearIsotropicSolid {
public:
  /**
   * @brief Construct a new Linear Isotropic Elasticity object
   *
   * @param density Density of the material
   * @param shear_modulus Shear modulus of the material
   * @param bulk_modulus Bulk modulus of the material
   */
  LinearIsotropicSolid(double density = 1.0, double shear_modulus = 1.0, double bulk_modulus = 1.0)
      : density_(density), bulk_modulus_(bulk_modulus), shear_modulus_(shear_modulus)
  {
    SLIC_ERROR_ROOT_IF(shear_modulus_ < 0.0,
                       "Shear modulus must be positive in the linear isotropic elasticity material model.");

    SLIC_ERROR_ROOT_IF(density_ < 0.0, "Density must be positive in the linear isotropic elasticity material model.");

    SLIC_ERROR_ROOT_IF(bulk_modulus_ < 0.0,
                       "Bulk modulus must be positive in the linear isotropic elasticity material model.");

    double K             = bulk_modulus;
    double G             = shear_modulus;
    double poisson_ratio = (3 * K - 2 * G) / (6 * K + 2 * G);

    SLIC_ERROR_ROOT_IF(poisson_ratio < 0.0,
                       "Poisson ratio must be positive in the linear isotropic elasticity material model.");
  }

  /**
   * @brief Material response call for a linear isotropic solid
   *
   * @tparam DisplacementType Displacement type
   * @tparam DispGradType Displacement gradient type
   * @param displacement_grad Displacement gradient with respect to the reference configuration (displacement_grad)
   * @return The calculated material response (density, Kirchoff stress) for the material
   */
  template <typename DisplacementType, typename DispGradType>
  SERAC_HOST_DEVICE auto operator()(const tensor<double, dim>& /* x */, const DisplacementType& /* displacement */,
                                    const DispGradType& displacement_grad) const
  {
    auto I      = Identity<dim>();
    auto lambda = bulk_modulus_ - (2.0 / dim) * shear_modulus_;
    auto strain = 0.5 * (displacement_grad + transpose(displacement_grad));
    auto stress = lambda * tr(strain) * I + 2.0 * shear_modulus_ * strain;
    return MaterialResponse<double, DispGradType>{.density = density_, .stress = stress};
  }

private:
  /// Density
  double density_;

  /// Bulk modulus
  double bulk_modulus_;

  /// Shear modulus
  double shear_modulus_;
};

/**
 * @brief Neo-Hookean material model
 *
 * @tparam dim The spatial dimension of the mesh
 */
template <int dim>
class NeoHookeanSolid {
public:
  /**
   * @brief Construct a new Neo-Hookean object
   *
   * @param density Density of the material
   * @param shear_modulus Shear modulus of the material
   * @param bulk_modulus Bulk modulus of the material
   */
  NeoHookeanSolid(double density = 1.0, double shear_modulus = 1.0, double bulk_modulus = 1.0)
      : density_(density), bulk_modulus_(bulk_modulus), shear_modulus_(shear_modulus)
  {
    SLIC_ERROR_ROOT_IF(shear_modulus_ < 0.0, "Shear modulus must be positive in the neo-Hookean material model.");

    SLIC_ERROR_ROOT_IF(density_ < 0.0, "Density must be positive in the neo-Hookean material model.");

    SLIC_ERROR_ROOT_IF(bulk_modulus_ < 0.0, "Bulk modulus must be positive in the neo-Hookean material model.");

    double K             = bulk_modulus;
    double G             = shear_modulus;
    double poisson_ratio = (3 * K - 2 * G) / (6 * K + 2 * G);

    SLIC_ERROR_ROOT_IF(poisson_ratio < 0.0, "Poisson ratio must be positive in the neo-Hookean material model.");
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
  SERAC_HOST_DEVICE auto operator()(const tensor<double, dim>& /* x */, const DisplacementType& /* displacement */,
                                    const DispGradType& displacement_grad) const
  {
    auto I      = Identity<dim>();
    auto lambda = bulk_modulus_ - (2.0 / dim) * shear_modulus_;
    auto B_minus_I =
        displacement_grad * transpose(displacement_grad) + transpose(displacement_grad) + displacement_grad;

    auto J = det(displacement_grad + I);

    // TODO this resolve to the correct std implementation of log when J resolves to a pure double. It can
    // be removed by either putting the dual implementation of the global namespace or implementing a pure
    // double version there. More investigation into argument-dependent lookup is needed.
    using std::log;
    auto stress = lambda * log(J) * I + shear_modulus_ * B_minus_I;

    return MaterialResponse{density_, stress};
  }

private:
  /// Density
  double density_;

  /// Bulk modulus in the stress free configuration
  double bulk_modulus_;

  /// Shear modulus in the stress free configuration
  double shear_modulus_;
};

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

}  // namespace serac::solid_util

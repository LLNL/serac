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
#include "serac/physics/materials/solid_functional_material.hpp"

/// SolidFunctional helper data types
namespace serac::solid_util {

/**
 * @brief Linear isotropic elasticity material model
 *
 * @tparam dim Spatial dimension of the mesh
 */
template <int dim>
class ParameterizedLinearIsotropicSolid {
public:
  /**
   * @brief Construct a new Linear Isotropic Elasticity object
   *
   * @param density Density of the material
   * @param shear_modulus_offset Shear modulus offset of the material
   * @param bulk_modulus_offset Bulk modulus offset of the material
   */
  ParameterizedLinearIsotropicSolid(double density = 1.0, double shear_modulus_offset = 1.0,
                                    double bulk_modulus_offset = 1.0)
      : density_(density), bulk_modulus_offset_(bulk_modulus_offset), shear_modulus_offset_(shear_modulus_offset)
  {
    SLIC_ERROR_ROOT_IF(density_ < 0.0, "Density must be positive in the linear isotropic elasticity material model.");
  }

  /**
   * @brief Material response call for a linear isotropic solid
   *
   * @tparam DisplacementType Displacement type
   * @tparam DispGradType Displacement gradient type
   * @tparam BulkType Bulk modulus type
   * @tparam ShearType Shear modulus type
   * @param displacement_grad Displacement gradient with respect to the reference configuration (displacement_grad)
   * @param bulk_parameter The parameterized bulk modulus
   * @param shear_parameter The parameterized shear modulus
   * @return The calculated material response (density, Kirchoff stress) for the material
   */
  template <typename DisplacementType, typename DispGradType, typename BulkType, typename ShearType>
  SERAC_HOST_DEVICE auto operator()(const tensor<double, dim>& /* x */, const DisplacementType& /* displacement */,
                                    const DispGradType& displacement_grad, const BulkType& bulk_parameter,
                                    const ShearType& shear_parameter) const
  {
    auto bulk_modulus  = bulk_parameter + bulk_modulus_offset_;
    auto shear_modulus = shear_parameter + shear_modulus_offset_;

    auto I      = Identity<dim>();
    auto lambda = bulk_modulus - (2.0 / dim) * shear_modulus;
    auto strain = 0.5 * (displacement_grad + transpose(displacement_grad));
    auto stress = lambda * tr(strain) * I + 2.0 * shear_modulus * strain;

    return MaterialResponse{.density = density_, .stress = stress};
  }

  /**
   * @brief The number of parameters in the model
   *
   * @return The number of parameters in the model
   */
  static constexpr int numParameters() { return 2; }

private:
  /// Density
  double density_;

  /// Bulk modulus
  double bulk_modulus_offset_;

  /// Shear modulus
  double shear_modulus_offset_;
};

/**
 * @brief Neo-Hookean material model
 *
 * @tparam dim The spatial dimension of the mesh
 */
template <int dim>
class ParameterizedNeoHookeanSolid {
public:
  /**
   * @brief Construct a new Neo-Hookean object
   *
   * @param density Density of the material
   * @param shear_modulus_offset Shear modulus of the material
   * @param bulk_modulus_offset Bulk modulus of the material
   */
  ParameterizedNeoHookeanSolid(double density = 1.0, double shear_modulus_offset = 1.0,
                               double bulk_modulus_offset = 1.0)
      : density_(density), bulk_modulus_offset_(bulk_modulus_offset), shear_modulus_offset_(shear_modulus_offset)
  {
    SLIC_ERROR_ROOT_IF(density_ < 0.0, "Density must be positive in the neo-Hookean material model.");
  }

  /**
   * @brief Material response call for a neo-Hookean solid
   *
   * @tparam DisplacementType Displacement type
   * @tparam DispGradType Displacement gradient type
   * @tparam BulkType Bulk modulus type
   * @tparam ShearType Shear modulus type
   * @param displacement_grad Displacement gradient with respect to the reference configuration (displacement_grad)
   * @param bulk_parameter The parameterized bulk modulus
   * @param shear_parameter The parameterized shear modulus
   * @return The calculated material response (density, kirchoff stress) for the material
   */
  template <typename DisplacementType, typename DispGradType, typename BulkType, typename ShearType>
  SERAC_HOST_DEVICE auto operator()(const tensor<double, dim>& /* x */, const DisplacementType& /* displacement */,
                                    const DispGradType& displacement_grad, const BulkType& bulk_parameter,
                                    const ShearType& shear_parameter) const
  {
    auto bulk_modulus  = bulk_parameter + bulk_modulus_offset_;
    auto shear_modulus = shear_parameter + shear_modulus_offset_;

    auto I      = Identity<dim>();
    auto lambda = bulk_modulus - (2.0 / dim) * shear_modulus;
    auto B_minus_I =
        displacement_grad * transpose(displacement_grad) + transpose(displacement_grad) + displacement_grad;

    auto J = det(displacement_grad + I);

    // TODO this resolve to the correct std implementation of log when J resolves to a pure double. It can
    // be removed by either putting the dual implementation of the global namespace or implementing a pure
    // double version there. More investigation into argument-dependent lookup is needed.
    using std::log;
    auto stress = lambda * log(J) * I + shear_modulus * B_minus_I;

    // TODO For some reason, clang can't handle CTAD here. We should investigate why.
    using StressType = decltype(stress);
    return MaterialResponse<double, StressType>{.density = density_, .stress = stress};
  }

  /**
   * @brief The number of parameters in the model
   *
   * @return The number of parameters in the model
   */
  static constexpr int numParameters() { return 2; }

private:
  /// Density
  double density_;

  /// Bulk modulus in the stress free configuration
  double bulk_modulus_offset_;

  /// Shear modulus in the stress free configuration
  double shear_modulus_offset_;
};

}  // namespace serac::solid_util

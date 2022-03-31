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
   * @tparam T1 Spatial position type
   * @tparam T2 Displacement type
   * @tparam T3 Displacement gradient type
   * @tparam T4 Bulk modulus type
   * @tparam T5 Shear modulus type
   * @param du_dX Displacement gradient with respect to the reference configuration (du_dX)
   * @param bulk_parameter The parameterized bulk modulus
   * @param shear_parameter The parameterized shear modulus
   * @return The calculated material response (density, kirchoff stress) for the material
   */
  template <typename T1, typename T2, typename T3, typename T4, typename T5>
  SERAC_HOST_DEVICE auto operator()(const T1& /* x */, const T2& /* displacement */, const T3& du_dX,
                                    const T4& bulk_parameter, const T5& shear_parameter) const
  {
    auto bulk_modulus  = bulk_parameter + bulk_modulus_offset_;
    auto shear_modulus = shear_parameter + shear_modulus_offset_;

    auto I      = Identity<dim>();
    auto lambda = bulk_modulus - (2.0 / dim) * shear_modulus;
    auto strain = 0.5 * (du_dX + transpose(du_dX));
    auto stress = lambda * tr(strain) * I + 2.0 * shear_modulus * strain;

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
   * @tparam T1 Spatial position type
   * @tparam T2 Displacement type
   * @tparam T3 Displacement gradient type
   * @tparam T4 Bulk modulus type
   * @tparam T5 Shear modulus type
   * @param du_dX Displacement gradient with respect to the reference configuration (du_dX)
   * @param bulk_parameter The parameterized bulk modulus
   * @param shear_parameter The parameterized shear modulus
   * @return The calculated material response (density, kirchoff stress) for the material
   */
  template <typename T1, typename T2, typename T3, typename T4, typename T5>
  SERAC_HOST_DEVICE auto operator()(const T1& /* x */, const T2& /* displacement */, const T3& du_dX,
                                    const T4& bulk_parameter, const T5& shear_parameter) const
  {
    auto bulk_modulus  = bulk_parameter + bulk_modulus_offset_;
    auto shear_modulus = shear_parameter + shear_modulus_offset_;

    auto I         = Identity<dim>();
    auto lambda    = bulk_modulus - (2.0 / dim) * shear_modulus;
    auto B_minus_I = du_dX * transpose(du_dX) + transpose(du_dX) + du_dX;

    auto J = det(du_dX + I);

    // TODO this resolve to the correct std implementation of log when J resolves to a pure double. It can
    // be removed by either putting the dual implementation of the global namespace or implementing a pure
    // double version there. More investigation into argument-dependent lookup is needed.
    using std::log;
    auto stress = lambda * log(J) * I + shear_modulus * B_minus_I;

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

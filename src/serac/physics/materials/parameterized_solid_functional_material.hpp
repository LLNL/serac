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
struct ParameterizedLinearIsotropicSolid {

  /**
   * @brief stress calculation for a linear isotropic material model
   *
   * @tparam DispGradType Displacement gradient type
   * @tparam BulkType Bulk modulus type
   * @tparam ShearType Shear modulus type
   * @param displacement_grad Displacement gradient with respect to the reference configuration (displacement_grad)
   * @param bulk_parameter The parameterized bulk modulus
   * @param shear_parameter The parameterized shear modulus
   * @return The calculated material response (density, Kirchoff stress) for the material
   */
  template <typename DisplacementType, typename DispGradType, typename BulkType, typename ShearType>
  SERAC_HOST_DEVICE auto operator()(const DispGradType& du_dX, const BulkType& DeltaK,
                                    const ShearType& DeltaG) const
  {
    constexpr auto I = Identity<dim>();
    auto K = K0 + DeltaK;
    auto G = G0 + DeltaG;
    auto lambda = K - (2.0 / dim) * G;
    auto epsilon = 0.5 * (transpose(du_dX) + du_dX);
    return lambda * tr(epsilon) * I + 2.0 * G * epsilon;
  }

  /**
   * @brief The number of parameters in the model
   *
   * @return The number of parameters in the model
   */
  static constexpr int numParameters() { return 2; }

  double density;
  double K0;
  double G0;
};

/**
 * @brief Neo-Hookean material model
 *
 * @tparam dim The spatial dimension of the mesh
 */
template <int dim>
struct ParameterizedNeoHookeanSolid {
  using State = Empty;

  /**
   * @brief stress calculation for a NeoHookean material model
   *
   * @tparam DispGradType Displacement gradient type
   * @tparam BulkType Bulk modulus type
   * @tparam ShearType Shear modulus type
   * @param du_dX Displacement gradient with respect to the reference configuration (displacement_grad)
   * @param bulk_parameter The parameterized bulk modulus
   * @param shear_parameter The parameterized shear modulus
   * @return The calculated material response (density, kirchoff stress) for the material
   */

  template <typename DispGradType, typename BulkType, typename ShearType>
  SERAC_HOST_DEVICE auto operator()(State & /*state*/, const DispGradType& du_dX, const BulkType& DeltaK, const ShearType& DeltaG) const
  {
    constexpr auto I = Identity<dim>();
    auto K = K0 + DeltaK;
    auto G = G0 + DeltaG;
    auto lambda = K - (2.0 / dim) * G;
    auto B_minus_I = du_dX * transpose(du_dX) + transpose(du_dX) + du_dX;
    return lambda * log(det(I + du_dX)) * I + G * B_minus_I;
  }

  /**
   * @brief The number of parameters in the model
   *
   * @return The number of parameters in the model
   */
  static constexpr int numParameters() { return 2; }

  double density;
  double K0;
  double G0;
};

}  // namespace serac::solid_util

// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermal_functional_material.hpp
 *
 * @brief The material and load types for the thermal functional physics module
 */

#pragma once

#include "serac/numerics/functional/functional.hpp"

/// ThermalConductionFunctional helper structs
namespace serac::Thermal {

/// Linear isotropic thermal conduction material model
struct LinearIsotropicConductor {
  /// Density
  double rho;

  /// Specific heat capacity
  double cp;

  /// Constant isotropic thermal conductivity
  double kappa;

  /**
   * @brief Function defining the thermal flux
   *
   * @tparam T1 type of the temperature (e.g. tensor or dual type)
   * @tparam T2 type of the temperature gradient (e.g. tensor or dual type)
   * @param du_dx Gradient of the temperature
   * @return The thermal flux of the material model
   */
  template <typename T1, typename T2>
  SERAC_HOST_DEVICE T2 operator()(T1& /* u */, T2& du_dx) const
  {
    return kappa * du_dx;
  }
};

// Use SFINAE to add static assertions checking if the given solid material type is acceptable
template <typename T, typename = void>
struct has_rho : std::false_type {
};

template <typename T>
struct has_rho<T, std::void_t<decltype(std::declval<T&>().rho)>> : std::true_type {
};

template <typename T, typename = void>
struct has_cp : std::false_type {
};

template <typename T>
struct has_cp<T, std::void_t<decltype(std::declval<T&>().cp)>> : std::true_type {
};

template <typename T, typename = void>
struct has_thermal_flux : std::false_type {
};

template <typename T>
struct has_thermal_flux<T, std::void_t<decltype(std::declval<T&>()(tensor<double, 1>{}, tensor<double, 3>{}))>>
    : std::true_type {
};

/// Constant thermal source model
struct ConstantSource {
  /// The constant source
  double source;

  /**
   * @brief Evaluation function for the constant thermal source model
   *
   * @tparam T1 type of the physical position
   * @tparam T2 type of the temperature
   * @tparam T3 type of the temperature gradient
   * @return The thermal source value
   */
  template <typename T1, typename T2, typename T3>
  SERAC_HOST_DEVICE T2 operator()(T1& /* x */, double /* t */, T2& u, T3& /* du_dx */) const
  {
    return source + u * 0.0;
  }
};

/// Constant thermal flux boundary model
struct FluxBoundary {
  /// The constant flux applied to the boundary
  double flux;

  /**
   * @brief Evaluation function for the thermal flux on a boundary
   *
   * @tparam T1 Type of the physical position
   * @tparam T2 Type of the normal vector
   * @tparam T3 Type of the temperature
   * @return The flux applied to the boundary
   */
  template <typename T1, typename T2, typename T3>
  SERAC_HOST_DEVICE T3 operator()(T1& /* x */, T2& /* n */, T3& u) const
  {
    return flux + u * 0.0;
  }
};

}  // namespace serac::Thermal
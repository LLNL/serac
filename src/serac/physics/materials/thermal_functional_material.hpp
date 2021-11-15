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
  double density_ = -1.0;

  /// Specific heat capacity
  double specific_heat_capacity_ = -1.0;

  /// Constant isotropic thermal conductivity
  double conductivity_ = -1.0;

  /**
   * @brief Function defining the thermal flux (constitutive response)
   *
   * @tparam T1 type of the temperature (e.g. tensor or dual type)
   * @tparam T2 type of the temperature gradient (e.g. tensor or dual type)
   * @param temperature_gradient Gradient of the temperature (du_dx)
   * @return The thermal flux of the material model
   */
  template <typename T1, typename T2>
  SERAC_HOST_DEVICE T2 operator()(T1& /* temperature */, T2& temperature_gradient) const
  {
    SLIC_ASSERT_MSG(conductivity_ > 0.0,
                    "Conductivity must be positive in the linear isotropic conductor material model.");

    return -1.0 * conductivity_ * temperature_gradient;
  }

  /**
   * @brief The density (mass per volume) of the material model
   *
   * @tparam T1 type of the position variable
   * @return The density
   */
  template <typename T1>
  SERAC_HOST_DEVICE double density(T1& /* x */) const
  {
    SLIC_ASSERT_MSG(density_ > 0.0, "Density must be positive in the linear isotropic conductor material model.");

    return density_;
  }

  /**
   * @brief The specific heat capacity (heat capacity per unit mass) of the material model
   *
   * @tparam T1 Type of the position variable
   * @tparam T2 Type of the temperature variable
   */
  template <typename T1, typename T2>
  SERAC_HOST_DEVICE double specificHeatCapacity(T1& /* x */, T2& /* temperature */) const
  {
    SLIC_ASSERT_MSG(specific_heat_capacity_ > 0.0,
                    "Specific heat capacity must be positive in the linear isotropic conductor material model.");

    return specific_heat_capacity_;
  }
};

/**
 * @brief Linear anisotropic thermal material model
 *
 * @tparam dim Spatial dimension
 */
template <int dim>
struct LinearConductor {
  /// Density
  double density_ = -1.0;

  /// Specific heat capacity
  double specific_heat_capacity_ = -1.0;

  /// Constant isotropic thermal conductivity
  tensor<double, dim, dim> conductivity_ = {-1.0};

  /**
   * @brief Function defining the thermal flux (constitutive response)
   *
   * @tparam T1 type of the temperature (e.g. tensor or dual type)
   * @tparam T2 type of the temperature gradient (e.g. tensor or dual type)
   * @param temperature_gradient Gradient of the temperature (du_dx)
   * @return The thermal flux of the material model
   */
  template <typename T1, typename T2>
  SERAC_HOST_DEVICE auto operator()(T1& /* temperature */, T2& temperature_gradient) const
  {
    for (int i = 0; i < dim; ++i) {
      SLIC_ASSERT_MSG(conductivity_(i, i) > 0.0,
                      "Conductivity tensor must be positive definite in linear conductor material model.");
    }
    return -1.0 * conductivity_ * temperature_gradient;
  }

  /**
   * @brief The density (mass per volume) of the material model
   *
   * @tparam T1 type of the position variable
   * @return The density
   */
  template <typename T1>
  SERAC_HOST_DEVICE double density(T1& /* x */) const
  {
    SLIC_ASSERT_MSG(density_ > 0.0, "Density must be positive in the linear conductor material model.");

    return density_;
  }

  /**
   * @brief The specific heat capacity (heat capacity per unit mass) of the material model
   *
   * @tparam T1 Type of the position variable
   * @tparam T2 Type of the temperature variable
   */
  template <typename T1, typename T2>
  SERAC_HOST_DEVICE double specificHeatCapacity(T1& /* x */, T2& /* temperature */) const
  {
    SLIC_ASSERT_MSG(specific_heat_capacity_ > 0.0,
                    "Specific heat capacity must be positive in the linear conductor material model.");

    return specific_heat_capacity_;
  }
};

// Use SFINAE to add static assertions checking if the given thermal material type is acceptable
template <typename T, int dim, typename = void>
struct has_density : std::false_type {
};

template <typename T, int dim>
struct has_density<T, dim, std::void_t<decltype(std::declval<T&>().density(std::declval<tensor<double, dim>&>()))>>
    : std::true_type {
};

template <typename T, int dim, typename = void>
struct has_specific_heat_capacity : std::false_type {
};

template <typename T, int dim>
struct has_specific_heat_capacity<T, dim,
                                  std::void_t<decltype(std::declval<T&>().specificHeatCapacity(
                                      std::declval<tensor<double, dim>&>(), std::declval<tensor<double, 1>&>()))>>
    : std::true_type {
};

template <typename T, int dim, typename = void>
struct has_thermal_flux : std::false_type {
};

template <typename T, int dim>
struct has_thermal_flux<
    T, dim,
    std::void_t<decltype(std::declval<T&>()(std::declval<tensor<double, 1>&>(), std::declval<tensor<double, dim>&>()))>>
    : std::true_type {
};

/// Constant thermal source model
struct ConstantSource {
  /// The constant source
  double source_;

  /**
   * @brief Evaluation function for the constant thermal source model
   *
   * @tparam T1 type of the physical position
   * @tparam T2 type of the temperature
   * @tparam T3 type of the temperature gradient
   * @return The thermal source value
   */
  template <typename T1, typename T2, typename T3>
  SERAC_HOST_DEVICE double operator()(T1& /* x */, double /* t */, T2& /* u */, T3& /* du_dx */) const
  {
    return source_;
  }
};

// Use SFINAE to add static assertions checking if the given thermal source type is acceptable
template <typename T, int dim, typename = void>
struct has_thermal_source : std::false_type {
};

template <typename T, int dim>
struct has_thermal_source<
    T, dim,
    std::void_t<decltype(std::declval<T&>()(std::declval<tensor<double, dim>&>(), std::declval<double>(),
                                            std::declval<tensor<double, 1>&>(), std::declval<tensor<double, dim>&>()))>>
    : std::true_type {
};

/// Constant thermal flux boundary model
struct FluxBoundary {
  /// The constant flux applied to the boundary
  double flux_;

  /**
   * @brief Evaluation function for the thermal flux on a boundary
   *
   * @tparam T1 Type of the physical position
   * @tparam T2 Type of the normal vector
   * @tparam T3 Type of the temperature
   * @return The flux applied to the boundary
   */
  template <typename T1, typename T2, typename T3>
  SERAC_HOST_DEVICE double operator()(T1& /* x */, T2& /* n */, T3& /* u */) const
  {
    return flux_;
  }
};

// Use SFINAE to add static assertions checking if the given thermal flux boundary type is acceptable
template <typename T, int dim, typename = void>
struct has_thermal_flux_boundary : std::false_type {
};

template <typename T, int dim>
struct has_thermal_flux_boundary<
    T, dim,
    std::void_t<decltype(std::declval<T&>()(std::declval<tensor<double, dim>&>(), std::declval<tensor<double, dim>&>(),
                                            std::declval<tensor<double, 1>&>()))>> : std::true_type {
};

}  // namespace serac::Thermal

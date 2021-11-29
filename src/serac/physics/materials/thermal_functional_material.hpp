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
class LinearIsotropicConductor {
public:
  /**
   * @brief Construct a new Linear Isotropic Conductor object
   *
   * @param density Density of the material (mass/volume)
   * @param specific_heat_capacity Specific heat capacity of the material (energy / (mass * temp))
   * @param conductivity Thermal conductivity of the material (power / (length * temp))
   */
  LinearIsotropicConductor(double density = 1.0, double specific_heat_capacity = 1.0, double conductivity = 1.0)
      : density_(density), specific_heat_capacity_(specific_heat_capacity), conductivity_(conductivity)
  {
    SLIC_ERROR_ROOT_IF(conductivity_ < 0.0,
                       "Conductivity must be positive in the linear isotropic conductor material model.");

    SLIC_ERROR_ROOT_IF(density_ < 0.0, "Density must be positive in the linear isotropic conductor material model.");

    SLIC_ERROR_ROOT_IF(specific_heat_capacity_ < 0.0,
                       "Specific heat capacity must be positive in the linear isotropic conductor material model.");
  }

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
    return -1.0 * conductivity_ * temperature_gradient;
  }

  /**
   * @brief The density (mass per volume) of the material model
   *
   * @tparam dim The dimension of the problem
   * @return The density
   */
  template <int dim>
  SERAC_HOST_DEVICE double density(tensor<double, dim>& /* x */) const
  {
    return density_;
  }

  /**
   * @brief The specific heat capacity (heat capacity per unit mass) of the material model
   *
   * @tparam dim The dimension of the problem
   * @tparam T Type of the temperature variable
   */
  template <typename T, int dim>
  SERAC_HOST_DEVICE double specificHeatCapacity(tensor<double, dim>& /* x */, T& /* temperature */) const
  {
    return specific_heat_capacity_;
  }

private:
  /// Density
  double density_ = 1.0;

  /// Specific heat capacity
  double specific_heat_capacity_ = 1.0;

  /// Constant isotropic thermal conductivity
  double conductivity_ = 1.0;
};

/**
 * @brief Linear anisotropic thermal material model
 *
 * @tparam dim Spatial dimension
 */
template <int dim>
class LinearConductor {
public:
  /**
   * @brief Construct a new Linear Isotropic Conductor object
   *
   * @param density Density of the material (mass/volume)
   * @param specific_heat_capacity Specific heat capacity of the material (energy / (mass * temp))
   * @param conductivity Thermal conductivity of the material (power / (length * temp))
   */
  LinearConductor(double density = 1.0, double specific_heat_capacity = 1.0,
                  tensor<double, dim, dim> conductivity = {{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}})
      : density_(density), specific_heat_capacity_(specific_heat_capacity), conductivity_(conductivity)
  {
    SLIC_ERROR_ROOT_IF(density_ < 0.0, "Density must be positive in the linear conductor material model.");

    SLIC_ERROR_ROOT_IF(specific_heat_capacity_ < 0.0,
                       "Specific heat capacity must be positive in the linear conductor material model.");

    // Check that the conductivity tensor is symmetric
    for (int i = 0; i < dim; ++i) {
      for (int j = i + 1; j < dim; ++j) {
        SLIC_ERROR_ROOT_IF(std::abs(conductivity_(i, j) - conductivity_(j, i)) > 1.0e-7,
                           "Conductivity tensor must be symmetric for the linear conductor material model.");
      }
    }

    // Check for positive definite conductivity using Sylvester's criterion
    // The upper left corner sub-matrices must have a positive determinant
    SLIC_ERROR_ROOT_IF(conductivity_(0, 0) < 0.0,
                       "Conductivity tensor must be positive definite for the linear conductor material model.");

    SLIC_ERROR_ROOT_IF(det(conductivity_) < 0.0,
                       "Conductivity tensor must be positive definite for the linear conductor material model.");

    if (dim == 3) {
      auto subtensor_2D = make_tensor<2, 2>([this](int i, int j) { return conductivity_(i, j); });
      SLIC_ERROR_ROOT_IF(det(subtensor_2D) < 0.0,
                         "Conductivity tensor must be positive definite for the linear conductor material model.");
    }
  }

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
    return -1.0 * conductivity_ * temperature_gradient;
  }

  /**
   * @brief The density (mass per volume) of the material model
   *
   * @return The density
   */
  SERAC_HOST_DEVICE double density(tensor<double, dim>& /* x */) const { return density_; }

  /**
   * @brief The specific heat capacity (heat capacity per unit mass) of the material model
   *
   * @tparam T Type of the temperature variable
   */
  template <typename T>
  SERAC_HOST_DEVICE double specificHeatCapacity(tensor<double, dim>& /* x */, T& /* temperature */) const
  {
    return specific_heat_capacity_;
  }

private:
  /// Density
  double density_ = 1.0;

  /// Specific heat capacity
  double specific_heat_capacity_ = 1.0;

  /// Constant thermal conductivity
  tensor<double, dim, dim> conductivity_ = {{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
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
   * @tparam T1 type of the temperature
   * @tparam T2 type of the temperature gradient
   * @tparam dim The dimension of the problem
   * @return The thermal source value
   */
  template <typename T1, typename T2, int dim>
  SERAC_HOST_DEVICE double operator()(tensor<double, dim>& /* x */, double /* t */, T1& /* u */, T2& /* du_dx */) const
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
   * @tparam T1 Type of the normal vector
   * @tparam T2 Type of the temperature
   * @tparam dim The dimension of the problem
   * @return The flux applied to the boundary
   */
  template <typename T1, typename T2, int dim>
  SERAC_HOST_DEVICE double operator()(tensor<double, dim>& /* x */, T1& /* n */, T2& /* u */) const
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

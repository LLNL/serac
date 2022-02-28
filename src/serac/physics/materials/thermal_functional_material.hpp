// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
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
  SERAC_HOST_DEVICE T2 operator()(const T1& /* temperature */, const T2& temperature_gradient) const
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
  SERAC_HOST_DEVICE double density(const tensor<double, dim>& /* x */) const
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
  SERAC_HOST_DEVICE double specificHeatCapacity(const tensor<double, dim>& /* x */, const T& /* temperature */) const
  {
    return specific_heat_capacity_;
  }

private:
  /// Density
  double density_;

  /// Specific heat capacity
  double specific_heat_capacity_;

  /// Constant isotropic thermal conductivity
  double conductivity_;
};

/// Linear isotropic thermal conduction material model
class ParameterizedLinearIsotropicConductor {
public:
  /**
   * @brief Construct a new Parameterized Linear Isotropic Conductor object
   *
   * @param density Density of the material (mass/volume)
   * @param specific_heat_capacity Specific heat capacity of the material (energy / (mass * temp))
   * @param conductivity Thermal conductivity of the material (power / (length * temp))
   */
  ParameterizedLinearIsotropicConductor(double density = 1.0, double specific_heat_capacity = 1.0,
                                        double conductivity = 1.0)
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
  template <typename T1, typename T2, typename T3>
  SERAC_HOST_DEVICE auto operator()(const T1& /* temperature */, const T2& temperature_gradient,
                                    const T3& parameter) const
  {
    return -1.0 * (conductivity_ + 0.01 * parameter) * temperature_gradient;
  }

  /**
   * @brief The density (mass per volume) of the material model
   *
   * @tparam dim The dimension of the problem
   * @return The density
   */
  template <int dim, typename T1>
  SERAC_HOST_DEVICE T1 density(const tensor<double, dim>& /* x */, const T1& parameter) const
  {
    return density_ + 0.0 * parameter;
  }

  /**
   * @brief The specific heat capacity (heat capacity per unit mass) of the material model
   *
   * @tparam dim The dimension of the problem
   * @tparam T Type of the temperature variable
   */
  template <typename T1, typename T2, int dim>
  SERAC_HOST_DEVICE double specificHeatCapacity(const tensor<double, dim>& /* x */, const T1& /* temperature */,
                                                const T2&) const
  {
    return specific_heat_capacity_;
  }

  static constexpr int numParameters() { return 1; }

private:
  /// Density
  double density_;

  /// Specific heat capacity
  double specific_heat_capacity_;

  /// Constant isotropic thermal conductivity
  double conductivity_;
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
                  tensor<double, dim, dim> conductivity = Identity<dim>())
      : density_(density), specific_heat_capacity_(specific_heat_capacity), conductivity_(conductivity)
  {
    SLIC_ERROR_ROOT_IF(density_ < 0.0, "Density must be positive in the linear conductor material model.");

    SLIC_ERROR_ROOT_IF(specific_heat_capacity_ < 0.0,
                       "Specific heat capacity must be positive in the linear conductor material model.");

    SLIC_ERROR_ROOT_IF(!is_symmetric_and_positive_definite(conductivity_),
                       "Conductivity tensor must be symmetric and positive definite.");
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
  SERAC_HOST_DEVICE auto operator()(const T1& /* temperature */, const T2& temperature_gradient) const
  {
    return -1.0 * conductivity_ * temperature_gradient;
  }

  /**
   * @brief The density (mass per volume) of the material model
   *
   * @return The density
   */
  SERAC_HOST_DEVICE double density(const tensor<double, dim>& /* x */) const { return density_; }

  /**
   * @brief The specific heat capacity (heat capacity per unit mass) of the material model
   *
   * @tparam T Type of the temperature variable
   */
  template <typename T>
  SERAC_HOST_DEVICE double specificHeatCapacity(const tensor<double, dim>& /* x */, const T& /* temperature */) const
  {
    return specific_heat_capacity_;
  }

private:
  /// Density
  double density_;

  /// Specific heat capacity
  double specific_heat_capacity_;

  /// Constant thermal conductivity
  tensor<double, dim, dim> conductivity_;
};

/// Constant thermal source model
struct ConstantSource {
  /// The constant source
  double source_ = 0.0;

  /**
   * @brief Evaluation function for the constant thermal source model
   *
   * @tparam T1 type of the temperature
   * @tparam T2 type of the temperature gradient
   * @tparam dim The dimension of the problem
   * @return The thermal source value
   */
  template <typename T1, typename T2, int dim>
  SERAC_HOST_DEVICE double operator()(const tensor<double, dim>& /* x */, const double /* t */, const T1& /* u */,
                                      const T2& /* du_dx */) const
  {
    return source_;
  }
};

/// Constant thermal flux boundary model
struct FluxBoundary {
  /// The constant flux applied to the boundary
  double flux_ = 0.0;

  /**
   * @brief Evaluation function for the thermal flux on a boundary
   *
   * @tparam T1 Type of the normal vector
   * @tparam T2 Type of the temperature
   * @tparam dim The dimension of the problem
   * @return The flux applied to the boundary
   */
  template <typename T1, typename T2, int dim>
  SERAC_HOST_DEVICE double operator()(const tensor<double, dim>& /* x */, const T1& /* n */, const T2& /* u */) const
  {
    return flux_;
  }
};

}  // namespace serac::Thermal

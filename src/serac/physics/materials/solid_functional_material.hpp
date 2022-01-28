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
namespace serac::Solid {

/// Linear isotropic thermal conduction material model
template <int dim>
class LinearIsotropicElasticity {
public:
  /**
   * @brief Construct a new Linear Isotropic Conductor object
   *
   * @param density Density of the material (mass/volume)
   * @param specific_heat_capacity Specific heat capacity of the material (energy / (mass * temp))
   * @param conductivity Thermal conductivity of the material (power / (length * temp))
   */
  LinearIsotropicElasticity(double density = 1.0, double shear_modulus = 1.0, double bulk_modulus = 1.0)
      : density_(density), bulk_modulus_(bulk_modulus), shear_modulus_(shear_modulus)
  {
    SLIC_ERROR_ROOT_IF(shear_modulus_ < 0.0,
                       "Conductivity must be positive in the linear isotropic conductor material model.");

    SLIC_ERROR_ROOT_IF(density_ < 0.0, "Density must be positive in the linear isotropic conductor material model.");

    SLIC_ERROR_ROOT_IF(bulk_modulus_ < 0.0,
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
  template <typename T>
  SERAC_HOST_DEVICE T operator()(const T& du_dX) const
  {
    auto I      = Identity<dim>();
    auto lambda = bulk_modulus_ - (2.0 / dim) * shear_modulus_;
    auto strain = 0.5 * (du_dX + transpose(du_dX));
    auto stress = lambda * tr(strain) * I + 2.0 * shear_modulus_ * strain;
    return stress;
  }

  /**
   * @brief The density (mass per volume) of the material model
   *
   * @tparam dim The dimension of the problem
   * @return The density
   */
  SERAC_HOST_DEVICE double density(const tensor<double, dim>& /* x */) const { return density_; }

private:
  /// Density
  double density_;

  /// Specific heat capacity
  double bulk_modulus_;

  /// Constant isotropic thermal conductivity
  double shear_modulus_;
};

/// Constant thermal source model
template <int dim>
struct ConstantBodyForce {
  /// The constant source
  tensor<double, dim> force_;

  /**
   * @brief Evaluation function for the constant thermal source model
   *
   * @tparam T1 type of the temperature
   * @tparam T2 type of the temperature gradient
   * @tparam dim The dimension of the problem
   * @return The thermal source value
   */
  template <typename T1, typename T2>
  SERAC_HOST_DEVICE tensor<double, dim> operator()(const tensor<double, dim>& /* x */, const double /* t */,
                                                   const T1& /* u */, const T2& /* du_dX */) const
  {
    return force_;
  }
};

/// Constant thermal source model
template <int dim>
struct ConstantTraction {
  /// The constant source
  tensor<double, dim> traction_;

  /**
   * @brief Evaluation function for the constant thermal source model
   *
   * @tparam T1 type of the temperature
   * @tparam T2 type of the temperature gradient
   * @tparam dim The dimension of the problem
   * @return The thermal source value
   */
  template <typename T1, typename T2>
  SERAC_HOST_DEVICE tensor<double, dim> operator()(const tensor<double, dim>& /* x */, const double /* t */,
                                                   const T1& /* u */, const T2& /* du_dX */) const
  {
    return traction_;
  }
};

/// Constant thermal source model
template <int dim>
struct TractionFunction {
  /// The constant source
  std::function<tensor<double, dim>(const tensor<double, dim>&, const tensor<double, dim>&, const double t)>
      traction_func_;

  /**
   * @brief Evaluation function for the constant thermal source model
   *
   * @tparam T1 type of the temperature
   * @tparam T2 type of the temperature gradient
   * @tparam dim The dimension of the problem
   * @return The thermal source value
   */
  SERAC_HOST_DEVICE tensor<double, dim> operator()(const tensor<double, dim>& x, const tensor<double, dim>& n,
                                                   const double t) const
  {
    return traction_func_(x, n, t);
  }
};

/// Constant thermal source model
struct ConstantPressure {
  /// The constant source
  double pressure_;

  /**
   * @brief Evaluation function for the constant thermal source model
   *
   * @tparam T1 type of the temperature
   * @tparam T2 type of the temperature gradient
   * @tparam dim The dimension of the problem
   * @return The thermal source value
   */
  template <int dim>
  SERAC_HOST_DEVICE double operator()(const tensor<double, dim>& /* x */, const double /* t */) const
  {
    return pressure_;
  }
};

/// Constant thermal source model
template <int dim>
struct PressureFunction {
  /// The constant source
  std::function<double(const tensor<double, dim>&, const double)> pressure_func_;

  /**
   * @brief Evaluation function for the constant thermal source model
   *
   * @tparam T1 type of the temperature
   * @tparam T2 type of the temperature gradient
   * @tparam dim The dimension of the problem
   * @return The thermal source value
   */
  SERAC_HOST_DEVICE double operator()(const tensor<double, dim>& x, const double t) const
  {
    return pressure_func_(x, t);
  }
};

}  // namespace serac::Solid

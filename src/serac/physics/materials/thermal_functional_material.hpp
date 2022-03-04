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

template <typename T1, typename T2, typename T3>
struct MaterialResponse {
  T1 density;
  T2 specific_heat_capacity;
  T3 heat_flux;
};

template <typename T1, typename T2, typename T3>
MaterialResponse(T1, T2, T3) -> MaterialResponse<T1, T2, T3>;

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

  template <typename T1, typename T2, typename T3>
  SERAC_HOST_DEVICE auto operator()(const T1&, const T2&, const T3& temperature_gradient) const
  {
    return MaterialResponse{.density                = density_,
                            .specific_heat_capacity = specific_heat_capacity_,
                            .heat_flux              = -conductivity_ * temperature_gradient};
  }

private:
  /// Density
  double density_;

  /// Specific heat capacity
  double specific_heat_capacity_;

  /// Constant isotropic thermal conductivity
  double conductivity_;
};

class ParameterizedLinearIsotropicConductor {
public:
  ParameterizedLinearIsotropicConductor(double density = 1.0, double specific_heat_capacity = 1.0,
                                        double conductivity = 1.0)
      : density_(density), specific_heat_capacity_(specific_heat_capacity), conductivity_(conductivity)
  {
    assert(conductivity > 0.0);
    assert(density > 0.0);
    assert(specific_heat_capacity > 0.0);
  }

  template <typename T1, typename T2, typename T3, typename T4, typename T5>
  SERAC_HOST_DEVICE auto operator()(const T1&, const T2&, const T3& temperature_gradient, const T4& parameter_1,
                                    const T5& parameter_2) const
  {
    return MaterialResponse{.density                = density_ + 0.01 * parameter_1,
                            .specific_heat_capacity = specific_heat_capacity_,
                            .heat_flux              = -1.0 * (conductivity_ + parameter_2) * temperature_gradient};
  }

  static constexpr int numParameters() { return 2; }

private:
  double density_;
  double specific_heat_capacity_;
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

  template <typename T1, typename T2, typename T3>
  SERAC_HOST_DEVICE auto operator()(const T1&, const T2&, const T3& temperature_gradient) const
  {
    return MaterialResponse{.density                = density_,
                            .specific_heat_capacity = specific_heat_capacity_,
                            .heat_flux              = -1.0 * (conductivity_)*temperature_gradient};
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
   * @tparam T1 type of the position vector
   * @tparam T2 type of the temperature 
   * @tparam T3 type of the temperature gradient
   * @tparam dim The dimension of the problem
   * @return The thermal source value
   */
  template <typename T1, typename T2, typename T3>
  SERAC_HOST_DEVICE auto operator()(const T1& /* x */, const double /* t */, const T2& /* u */,
                                    const T3& /* du_dx */) const
  {
    return source_;
  }
};

/// Constant thermal flux boundary model
struct ConstantFlux {
  /// The constant flux applied to the boundary
  double flux_ = 0.0;

  /**
   * @brief Evaluation function for the thermal flux on a boundary
   *
   * @tparam T1 Type of the position vector
   * @tparam T2 Type of the normal vector
   * @tparam T3 Type of the temperature on the boundary
   * @tparam dim The dimension of the problem
   * @return The flux applied to the boundary
   */
  template <typename T1, typename T2, typename T3>
  SERAC_HOST_DEVICE auto operator()(const T1& /* x */, const T2& /* n */, const T3& /* u */) const
  {
    return flux_;
  }
};

}  // namespace serac::Thermal

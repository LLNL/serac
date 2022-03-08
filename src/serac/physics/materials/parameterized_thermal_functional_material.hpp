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

#include "serac/physics/materials/thermal_functional_material.hpp"

/// ThermalConductionFunctional helper structs
namespace serac::Thermal {

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

  template <typename T1, typename T2, typename T3, typename T4>
  SERAC_HOST_DEVICE auto operator()(const T1&, const T2&, const T3& temperature_gradient, const T4& parameter) const
  {
    return MaterialResponse{.density                = density_,
                            .specific_heat_capacity = specific_heat_capacity_,
                            .heat_flux              = -1.0 * (conductivity_ + parameter) * temperature_gradient};
  }

  static constexpr int numParameters() { return 1; }

private:
  double density_;
  double specific_heat_capacity_;
  double conductivity_;
};

/// Parameterized thermal source model
struct ParameterizedSource {
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
  template <typename T1, typename T2, typename T3, typename T4>
  SERAC_HOST_DEVICE auto operator()(const T1& /* x */, const double /* t */, const T2& /* u */, const T3& /* du_dx */,
                                    const T4& parameter) const
  {
    return source_ + parameter;
  }

  static constexpr int numParameters() { return 1; }
};

/// Constant thermal flux boundary model
struct ParameterizedFlux {
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
  template <typename T1, typename T2, typename T3, typename T4>
  SERAC_HOST_DEVICE auto operator()(const T1& /* x */, const T2& /* n */, const T3& /* u */, const T4& parameter) const
  {
    return flux_ + parameter;
  }

  static constexpr int numParameters() { return 1; }
};

}  // namespace serac::Thermal

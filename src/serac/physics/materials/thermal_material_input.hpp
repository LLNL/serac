// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermal_material_input.hpp
 *
 * @brief This file contains functions for reading a material from input files
 */

#pragma once

#include <string>
#include "serac/infrastructure/input.hpp"
#include "serac/physics/materials/thermal_material.hpp"

namespace serac {

/// @brief Holds all possible heat transfer materials that can be utilized in our Input Deck
using var_thermal_material_t = std::variant<heat_transfer::LinearIsotropicConductor, heat_transfer::LinearConductor<2>,
                                            heat_transfer::LinearConductor<3>>;

/// @brief Contains function that defines the schema for heat transfer materials
struct ThermalMaterialInputOptions {
  /**
   * @brief Input file parameters specific to this class
   *
   * @param[in] container Inlet's Container to which fields should be added
   **/
  static void defineInputFileSchema(axom::inlet::Container& container);
};

}  // namespace serac

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by Inlet
 */
template <>
struct FromInlet<std::vector<std::vector<double>>> {
  /// @brief Returns created object from Inlet container
  std::vector<std::vector<double>> operator()(const axom::inlet::Container& base);
};

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by Inlet
 */
template <>
struct FromInlet<serac::var_thermal_material_t> {
  /// @brief Returns created object from Inlet container
  serac::var_thermal_material_t operator()(const axom::inlet::Container& base);
};

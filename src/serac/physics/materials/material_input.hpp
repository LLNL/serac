// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file material_input.hpp
 *
 * @brief This file contains functions for reading a material from input files
 */

#pragma once

#include <string>
#include "axom/inlet.hpp"
#include "serac/infrastructure/input.hpp"

namespace serac {

/**
 * @brief The information required from the input file for a material. Contains material parameters for all materials.
 */
struct MaterialInputOptions {
  /**
   * @brief Input file parameters specific to this class
   *
   * @param[in] container Inlet's Container to which fields should be added
   **/
  static void defineInputFileSchema(axom::inlet::Container& container);

  /**
   * @brief Model of material
   * 
   */
  std::string model;

  /** 
   * @brief mass density
   *
  */
  double density;

  /**
   * @brief The shear modulus
   *
   */
  double mu;

  /**
   * @brief The bulk modulus
   *
   */
  double K;

  /** 
   * @brief Young's modulus
   *
  */
  double E;

  /** 
   * @brief Poisson's ratio
   *
  */
  double nu;

  /** 
   * @brief Isotropic hardening constant
   *
  */
  double Hi;

  /** 
   * @brief Kinematic hardening constant
   *
  */
  double Hk;

  /** 
   * @brief Yield stress
   *
  */
  double sigma_y;

  /**
   * @brief The conductivity parameter
   *
   */
  double kappa;

  /**
   * @brief The specific heat capacity
   *
   */
  double cp;
};

} // namespace serac

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by Inlet
 */
template <>
struct FromInlet<serac::MaterialInputOptions> {
  /// @brief Returns created object from Inlet container
  serac::MaterialInputOptions operator()(const axom::inlet::Container& base);
};

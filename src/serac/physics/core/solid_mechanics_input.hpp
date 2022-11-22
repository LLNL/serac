// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_mechanics_input.hpp
 *
 * @brief An object containing all input file options for the solver for 
 * total Lagrangian finite deformation solid mechanics
 */

#pragma once

#include "serac/infrastructure/input.hpp"
#include "serac/physics/core/common.hpp"

namespace serac {

/**
 * @brief Stores all information held in the input file that
 * is used to configure the solver
 */
struct SolidMechanicsInputOptions {
  /**
   * @brief Input file parameters specific to this class
   *
   * @param[in] container Inlet container on which the input schema will be defined
   **/
  static void defineInputFileSchema(axom::inlet::Container& container);

  /**
   * @brief The order of the discretization
   *
   */
  int order;

  /**
   * @brief The options for the linear, nonlinear, and ODE solvers
   *
   */
  SolverOptions solver_options;

  /**
   * @brief The shear modulus
   *
   */
  //TODO: Move to material options
  double mu;

  /**
   * @brief The bulk modulus
   *
   */
  //TODO: Move to material options
  double K;

  /**
   * @brief The linear viscosity coefficient
   *
   */
  double viscosity;

  /**
   * @brief Initial density
   *
   */
  double initial_mass_density;

  /**
   * @brief Geometric nonlinearities flag
   *
   */
  GeometricNonlinearities geom_nonlin;

  /**
   * @brief Material nonlinearities flag
   *
   */
  bool material_nonlin;

  /**
   * @brief Boundary condition information
   *
   */
  std::unordered_map<std::string, input::BoundaryConditionInputOptions> boundary_conditions;

  /**
   * @brief The initial displacement
   * @note This can be used as an initialization field for dynamic problems or an initial guess
   *       for quasi-static solves
   *
   */
  std::optional<input::CoefficientInputOptions> initial_displacement;

  /**
   * @brief The initial velocity
   *
   */
  std::optional<input::CoefficientInputOptions> initial_velocity;
};

}  // namespace serac

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by Inlet
 */
template <>
struct FromInlet<serac::SolidMechanicsInputOptions> {
  /// @brief Returns created object from Inlet container
  serac::SolidMechanicsInputOptions operator()(const axom::inlet::Container& base);
};

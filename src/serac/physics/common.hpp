// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file common.hpp
 *
 * @brief A file defining some enums and structs that are used by the different physics modules 
 */
#pragma once

#include "serac/numerics/odes.hpp"

namespace serac {

/**
 * @brief a struct that is used in the physics modules to clarify which template arguments are 
 * user-controlled parameters (e.g. for design optimization)
 */
template < typename ... T >
struct Parameters{
  static constexpr int n = sizeof ... (T); ///< how many parameters were specified
};

/// A timestep and boundary condition enforcement method for a dynamic solver
struct TimesteppingOptions {
  /// The timestepping method to be applied
  TimestepMethod timestepper;

  /// The essential boundary enforcement method to use
  DirichletEnforcementMethod enforcement_method;
};

/**
 * @brief A configuration variant for the various solves
 * For quasistatic solves, leave the @a dyn_options parameter null. @a T_nonlin_options and @a T_lin_options
 * define the solver parameters for the nonlinear residual and linear stiffness solves. For
 * dynamic problems, @a dyn_options defines the timestepping scheme while @a T_lin_options and @a T_nonlin_options
 * define the nonlinear residual and linear stiffness solve options as before.
 */
struct SolverOptions {

  /// the method, iteration limit, and tolerances for the linear system
  LinearSolverOptions linear;

  /// the method, iteration limit, and tolerances for the nonlinear system
  NonlinearSolverOptions nonlinear;

  /**
   * @brief The optional ODE solver parameters
   * @note If this is not defined, a quasi-static solve is performed
   */
  std::optional<TimesteppingOptions> dynamic = std::nullopt;
};

}

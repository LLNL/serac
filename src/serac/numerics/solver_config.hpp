// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solver_config.hpp
 *
 * @brief This file contains enumerations and record types for physics solver configuration
 */

#pragma once

#include <variant>

#include "mfem.hpp"

namespace serac {

/**
 * @brief Timestep method of a solver
 */
enum class TimestepMethod
{
  QuasiStatic, /**< Quasistatic */

  // options for first order ODEs
  BackwardEuler,    /**< FirstOrderODE option */
  SDIRK33,          /**< FirstOrderODE option */
  ForwardEuler,     /**< FirstOrderODE option */
  RK2,              /**< FirstOrderODE option */
  RK3SSP,           /**< FirstOrderODE option */
  RK4,              /**< FirstOrderODE option */
  GeneralizedAlpha, /**< FirstOrderODE option */
  ImplicitMidpoint, /**< FirstOrderODE option */
  SDIRK23,          /**< FirstOrderODE option */
  SDIRK34,          /**< FirstOrderODE option */

  // options for second order ODEs
  //
  // note: we don't have a way to communicate
  //       parameters to the TimestepMethod,
  //       right now, so Newmark implies
  //       (beta = 0.25, gamma = 0.5)
  Newmark,             /**< SecondOrderODE option */
  HHTAlpha,            /**< SecondOrderODE option */
  WBZAlpha,            /**< SecondOrderODE option */
  AverageAcceleration, /**< SecondOrderODE option */
  LinearAcceleration,  /**< SecondOrderODE option */
  CentralDifference,   /**< SecondOrderODE option */
  FoxGoodwin           /**< SecondOrderODE option */
};

/**
 * @brief this enum describes which way to enforce the time-varying constraint u(t) == U(t)
 */

enum class DirichletEnforcementMethod
{
  /**
   * Satisfies u(t+dt) == U(t+dt)
   *
   * This method imposes additional stability criteria
   * for the case of second order differential equations
   */
  DirectControl,

  /**
   * (default value)
   * Satisfies dudt(t+dt) == dUdt(t+dt)
   *
   * This method does not impose any additional stability criteria
   * for the case of second order differential equations.
   */
  RateControl,

  /**
   * satisfies u(t+dt) == U(t+dt),
   *           dudt(t+dt) == dUdt(t+dt),
   * (and      d2udt2(t+dt) == d2Udt2(t+dt), for a second order ODE)
   *
   * Empirically, this method tends to be the most accurate
   * for small timesteps (by a constant factor),  but is more
   * expensive to evaluate
   */
  FullControl
};

/// A timestep and boundary condition enforcement method for a dynamic solver
struct TimesteppingOptions {
  /// The timestepping method to be applied
  TimestepMethod timestepper = TimestepMethod::QuasiStatic;

  /// The essential boundary enforcement method to use
  DirichletEnforcementMethod enforcement_method = DirichletEnforcementMethod::RateControl;
};

// _linear_solvers_start
/// Linear solution method indicator
enum class LinearSolver
{
  CG,       /**< Conjugate gradient */
  GMRES,    /**< Generalized minimal residual method */
  SuperLU,  /**< SuperLU MPI-enabled direct nodal solver */
  Strumpack /**< Strumpack MPI-enabled direct frontal solver*/
};
// _linear_solvers_end

// Add a custom list of strings? conduit node?
// Arbitrary string (e.g. json) to define parameters?

// _nonlinear_solvers_start
/// Nonlinear solver method indicator
enum class NonlinearSolver
{
  Newton,           /**< MFEM-native Newton-Raphson */
  LBFGS,            /**< MFEM-native Limited memory BFGS */
  NewtonLineSearch, /**< Custom solver using preconditioned earch direction with backtracking line search */
  Nesterov,         /**< Custom solver using Nesterov dynamic (damped) dynamics for accelerating implicit solves */
  TrustRegion,      /**< Custom solver using a trust region solver */
  KINFullStep,      /**< KINSOL Full Newton (Sundials must be enabled) */
  KINBacktrackingLineSearch, /**< KINSOL Newton with Backtracking Line Search (Sundials must be enabled) */
  KINPicard                  /**< KINSOL Picard (Sundials must be enabled) */
};
// _nonlinear_solvers_end

/**
 * @brief Solver types supported by AMGX
 */
enum class AMGXSolver
{
  AMG,            /**< GPU Algebraic Multigrid */
  PCGF,           /**< GPU PCGF */
  CG,             /**< GPU CG */
  PCG,            /**< GPU PCG */
  PBICGSTAB,      /**< GPU PBICGSTAB */
  BICGSTAB,       /**< GPU BICGSTAB */
  FGMRES,         /**< GPU FGMRES */
  JACOBI_L1,      /**< GPU JACOBI_L1 */
  GS,             /**< GPU GS */
  POLYNOMIAL,     /**< GPU POLYNOMIAL */
  KPZ_POLYNOMIAL, /**< GPU KPZ_POLYNOMIAL */
  BLOCK_JACOBI,   /**< GPU BLOCK_JACOBI */
  MULTICOLOR_GS,  /**< GPU MULTICOLOR_GS */
  MULTICOLOR_DILU /**< GPU MULTICOLOR_DILU */
};

/**
 * @brief Stores the information required to configure a NVIDIA AMGX preconditioner
 */
struct AMGXOptions {
  /**
   * @brief The solver algorithm
   */
  AMGXSolver solver = AMGXSolver::AMG;
  /**
   * @brief The smoother algorithm
   */
  AMGXSolver smoother = AMGXSolver::JACOBI_L1;
  /**
   * @brief Whether to display statistics from AMGX
   */
  bool verbose = false;
};

// _preconditioners_start
/// The type of preconditioner to be used
enum class Preconditioner
{
  HypreJacobi,      /**< Hypre-based Jacobi */
  HypreL1Jacobi,    /**< Hypre-based L1-scaled Jacobi */
  HypreGaussSeidel, /**< Hypre-based Gauss-Seidel */
  HypreAMG,         /**< Hypre's BoomerAMG algebraic multi-grid */
  HypreILU,         /**< Hypre's Incomplete LU */
  AMGX,             /**< NVIDIA's AMGX GPU-enabled algebraic multi-grid, GPU builds only */
  None              /**< No preconditioner used */
};
// _preconditioners_end

// _linear_options_start
/// Parameters for an iterative linear solution scheme
struct LinearSolverOptions {
  /// Linear solver selection
  LinearSolver linear_solver = LinearSolver::GMRES;

  /// PreconditionerOptions selection
  Preconditioner preconditioner = Preconditioner::HypreJacobi;

  /// Relative tolerance
  double relative_tol = 1.0e-8;

  /// Absolute tolerance
  double absolute_tol = 1.0e-12;

  /// Maximum number of iterations
  int max_iterations = 300;

  /// Debugging print level for the linear solver
  int print_level = 0;

  /// Debugging print level for the preconditioner
  int preconditioner_print_level = 0;
};
// _linear_options_end

// _nonlinear_options_start
/// Nonlinear solution scheme parameters
struct NonlinearSolverOptions {
  /// Nonlinear solver selection
  NonlinearSolver nonlin_solver = NonlinearSolver::NewtonLineSearch;

  /// Relative tolerance
  double relative_tol = 1.0e-8;

  /// Absolute tolerance
  double absolute_tol = 1.0e-12;

  /// Minimum number of iterations
  int min_iterations = 0;

  /// Maximum number of iterations
  int max_iterations = 20;

  /// Maximum line search cutbacks
  int max_line_search_iterations = 0;

  /// Debug print level
  int print_level = 0;
};
// _nonlinear_options_end

}  // namespace serac

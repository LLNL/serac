// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solver_config.hpp
 *
 * @brief This file contains enumerations and record types for physics solver configuration
 */

#ifndef SOLVER_CONFIG
#define SOLVER_CONFIG

namespace serac {
/**
 * @brief Output file type associated with a solver
 */
enum class OutputType
{
  GLVis,
  VisIt
};

/**
 * @brief Timestep method of a solver
 */
enum class TimestepMethod
{
  BackwardEuler,
  SDIRK33,
  ForwardEuler,
  RK2,
  RK3SSP,
  RK4,
  GeneralizedAlpha,
  ImplicitMidpoint,
  SDIRK23,
  SDIRK34,
  QuasiStatic, 

  HHTAlpha,
  WBZAlpha,
  AverageAcceleration,
  LinearAcceleration,
  CentralDifference,
  FoxGoodwin
};

// this enum describes the which way to
// enforce the time-varying constraint 
//   u(t) == U(t)
enum class DirichletEnforcementMethod
{
  // satisfies u(t+dt) == U(t+dt)
  // 
  // this method imposes additional stability criteria 
  // for the case of second order differential equations
  DirectControl, 

  // (default value)
  // satisfies dudt(t+dt) == dUdt(t+dt)
  // 
  // this method does not impose any additional stability criteria 
  // for the case of second order differential equations.
  RateControl,   

  // satisfies u(t+dt) == U(t+dt), 
  //           dudt(t+dt) == dUdt(t+dt),
  // (and      d2udt2(t+dt) == d2Udt2(t+dt), for a second order ODE)
  // 
  // Empirically, this method tends to be the most accurate
  // for small timesteps (by a constant factor),  but is more 
  // expensive to evaluate 
  FullControl    
};

/**
 * @brief Linear solution method
 */
enum class LinearSolver
{
  CG,
  GMRES,
  MINRES,
  SuperLU
};

/**
 * @brief Preconditioning method
 */
enum class Preconditioner
{
  Jacobi,
  BoomerAMG
};

/**
 * @brief Abstract multiphysics coupling scheme
 */
enum class CouplingScheme
{
  OperatorSplit,
  FixedPoint,
  FullyCoupled
};

/**
 * @brief Parameters for a linear solution scheme
 */
struct LinearSolverParameters {
  /**
   * @brief Relative tolerance
   */
  double rel_tol;

  /**
   * @brief Absolute tolerance
   */
  double abs_tol;

  /**
   * @brief Debugging print level
   */
  int print_level;

  /**
   * @brief Maximum number of iterations
   */
  int max_iter;

  /**
   * @brief Linear solver selection
   */
  LinearSolver lin_solver;

  /**
   * @brief Preconditioner selection
   */
  Preconditioner prec;
};

/**
 * @brief Nonlinear solution scheme parameters
 */
struct NonlinearSolverParameters {
  /**
   * @brief Relative tolerance
   */
  double rel_tol;

  /**
   * @brief Absolute tolerance
   */
  double abs_tol;

  /**
   * @brief Maximum number of iterations
   */
  int max_iter;

  /**
   * @brief Debug print level
   */
  int print_level;
};

}  // namespace serac

#endif

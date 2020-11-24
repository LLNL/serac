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

#include <variant>

#include "mfem.hpp"

namespace serac {
/**
 * @brief Output file type associated with a solver
 */
enum class OutputType
{
  GLVis,
  ParaView,
  VisIt,
  SidreVisIt
};

/**
 * @brief Timestep method of a solver
 */
enum class TimestepMethod
{
  QuasiStatic,

  // options for first order ODEs
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

  // options for second order ODEs
  HHTAlpha,
  WBZAlpha,
  AverageAcceleration,
  LinearAcceleration,
  CentralDifference,
  FoxGoodwin
};

/**
 * @brief this enum describes which way to enforce the time-varying constraint u(t) == U(t)
 */

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
 * @brief Nonlinear solver type/method
 */
enum class NonlinearSolver
{
  MFEMNewton,
  KINFullStep,
  KINBacktrackingLineSearch
};

/**
 * @brief Stores the information required to configure a HypreSmoother
 */
struct HypreSmootherPrec {
  mfem::HypreSmoother::Type type;
};

/**
 * @brief Stores the information required to configure a HypreBoomerAMG preconditioner
 */
struct HypreBoomerAMGPrec {
  mfem::ParFiniteElementSpace* pfes = nullptr;
};

/**
 * @brief Stores the information required to configure a NVIDIA AMGX preconditioner
 */
struct AMGXPrec {
};

/**
 * @brief Stores the information required to configure a BlockILU preconditioner
 */
struct BlockILUPrec {
  int block_size;
};

/**
 * @brief Preconditioning method
 */
using Preconditioner = std::variant<HypreSmootherPrec, HypreBoomerAMGPrec, AMGXPrec, BlockILUPrec>;

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
 * @brief Parameters for an iterative linear solution scheme
 */
struct IterativeSolverParameters {
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
  std::optional<Preconditioner> prec;
};

/**
 * @brief Parameters for a custom solver (currently just a non-owning pointer to the solver)
 * @note This is preferable to unique_ptr or even references because non-trivial copy constructors
 * and destructors are a nightmare in this context
 */
struct CustomSolverParameters {
  mfem::Solver* solver = nullptr;
};

/**
 * @brief Parameters for a direct solver (PARDISO, MUMPS, SuperLU, etc)
 */
struct DirectSolverParameters {
  /**
   * @brief Debugging print level
   */
  int print_level;
};

/**
 * @brief Parameters for a linear solver
 */
using LinearSolverParameters = std::variant<IterativeSolverParameters, CustomSolverParameters, DirectSolverParameters>;

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

  /**
   * @brief Nonlinear solver selection
   */
  NonlinearSolver nonlin_solver = NonlinearSolver::MFEMNewton;
};

}  // namespace serac

#endif

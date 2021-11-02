// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
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
 * @brief Output file type associated with a solver
 */
enum class OutputType
{
  GLVis,     /**< GLVis output */
  ParaView,  /**< Paraview output */
  VisIt,     /**< VisIt output */
  SidreVisIt /**< Binary VisIt output via Sidre */
};

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

/**
 * @brief Linear solution method
 */
enum class LinearSolver
{
  CG,     /**< Conjugate Gradient */
  GMRES,  /**< Generalized minimal residual method */
  MINRES, /**< Minimal residual method */
  SuperLU /**< SuperLU Direct Solver */
};

/**
 * @brief Nonlinear solver type/method
 */
enum class NonlinearSolver
{
  MFEMNewton,               /**< Newton-Raphson */
  KINFullStep,              /**< KINFullStep */
  KINBacktrackingLineSearch /**< KINBacktrackingLineSearch */
};

/**
 * @brief Stores the information required to configure a HypreSmoother
 */
struct HypreSmootherPrec {
  /**
   * @brief The type of Hypre smoother to apply
   */
  mfem::HypreSmoother::Type type;
};

/**
 * @brief Stores the information required to configure a HypreBoomerAMG preconditioner
 */
struct HypreBoomerAMGPrec {
  /**
   * @brief The par finite element space for the AMG object
   * @note This is needed for some of the options specific to solid mechanics solves
   */
  mfem::ParFiniteElementSpace* pfes = nullptr;
};

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
struct AMGXPrec {
  /**
   * @brief The solver algorithm
   */
  AMGXSolver solver = AMGXSolver::AMG;
  /**
   * @brief The smoother algorithm
   */
  AMGXSolver smoother = AMGXSolver::BLOCK_JACOBI;
  /**
   * @brief Whether to display statistics from AMGX
   */
  bool verbose = false;
};

/**
 * @brief Stores the information required to configure a BlockILU preconditioner
 */
struct BlockILUPrec {
  /**
   * @brief The block size for the ILU preconditioner
   */
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
  OperatorSplit, /**< Operator Split */
  FixedPoint,    /**< Fixed Point */
  FullyCoupled   /**< FullyCoupled */
};

/**
 * @brief Parameters for an iterative linear solution scheme
 */
struct IterativeSolverOptions {
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
struct CustomSolverOptions {
  /**
   * @brief A non-owning pointer to the custom mfem solver to use
   */
  mfem::Solver* solver = nullptr;
};

/**
 * @brief Parameters for a direct solver (PARDISO, MUMPS, SuperLU, etc)
 */
struct DirectSolverOptions {
  /**
   * @brief Debugging print level
   */
  int print_level;
};

/**
 * @brief Parameters for a linear solver
 */
using LinearSolverOptions = std::variant<IterativeSolverOptions, CustomSolverOptions, DirectSolverOptions>;

/**
 * @brief Nonlinear solution scheme parameters
 */
struct NonlinearSolverOptions {
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

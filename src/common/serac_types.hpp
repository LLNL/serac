// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file serac_types.hpp
 *
 * @brief This file contains common serac data structures
 */

#ifndef SERAC_TYPES
#define SERAC_TYPES

#include <memory>

#include "common/logger.hpp"
#include "mfem.hpp"

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
  QuasiStatic
};

/**
 * @brief Linear solution method
 */
enum class LinearSolver
{
  CG,
  GMRES,
  MINRES
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

/**
 * @brief Bundle of data containing a complete finite element state variable
 */
struct FiniteElementState {
  /**
   * @brief Finite element space (basis functions and their transformations)
   */
  std::shared_ptr<mfem::ParFiniteElementSpace> space;

  /**
   * @brief Finite element collection (reference configuration basis functions)
   */
  std::shared_ptr<mfem::FiniteElementCollection> coll;

  /**
   * @brief Grid function (DOF vector and associated finite element space)
   */
  std::shared_ptr<mfem::ParGridFunction> gf;

  /**
   * @brief True vector (Non-constrained DOF vector)
   */
  std::shared_ptr<mfem::Vector> true_vec;

  /**
   * @brief The parallel mesh
   */
  std::shared_ptr<mfem::ParMesh> mesh;

  /**
   * @brief Name of the state variable
   */
  std::string name = "";
};

}  // namespace serac

#endif

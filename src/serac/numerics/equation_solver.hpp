// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file equation_solver.hpp
 *
 * @brief This file contains the declaration of an equation solver wrapper
 */

#pragma once

#include <memory>
#include <optional>
#include <variant>

#include "mfem.hpp"

#include "serac/infrastructure/input.hpp"
#include "serac/numerics/solver_config.hpp"

namespace serac::mfem_ext {

/**
 * @brief This class manages the objects typically required to solve a nonlinear set of equations arising from
 * discretization of a PDE. Specifically, it has
 *
 *   1. An @a mfem::NewtonSolver containing the nonlinear solution operator
 *   2. An optional @a mfem::Solver containing a linear solver that is used by the nonlinear solution operator
 *   3. An optional @a mfem::Solver containing a preconditioner for the linear solution operator
 *
 * This @a EquationSolver manages these objects together to ensure they all exist when called by their associated
 * physics simulation module.
 *
 * An equation solver can either be constructed by supplying pre-built nonlinear and linear solvers with a
 * preconditioner, or it can be constructed using @a serac::NonlinearSolverOptions and @a serac::LinearSolverOptions
 * structs with the
 * @ref serac::mfem_ext::buildEquationSolver factory method.
 */
class EquationSolver : public mfem::Solver {
public:
  /**
   * Constructs a new nonlinear equation solver
   * @param[in] nonlinear_solver A constructed nonlinear solver
   * @param[in] linear_solver An optional constructed linear solver to be called by the nonlinear algorithm and adjoint
   * equation solves
   * @param[in] preconditioner An optional constructed precondition to aid the linear solver
   */
  EquationSolver(std::unique_ptr<mfem::NewtonSolver> nonlinear_solver,
                 std::unique_ptr<mfem::Solver>       linear_solver  = nullptr,
                 std::unique_ptr<mfem::Solver>       preconditioner = nullptr);

  /**
   * Updates the solver with the provided operator
   * @param[in] op The operator (system matrix) to use, "A" in Ax = b
   * @note Implements mfem::Operator::SetOperator
   */
  void SetOperator(const mfem::Operator& op) override;

  /**
   * Solves the system
   * @param[in] b RHS of the system of equations
   * @param[out] x Solution to the system of equations
   * @note Implements mfem::Operator::Mult
   */
  void Mult(const mfem::Vector& b, mfem::Vector& x) const override;

  /**
   * Returns the underlying solver object
   * @return A non-owning reference to the underlying nonlinear solver
   */
  mfem::Solver* NonlinearSolver() { return nonlin_solver_.get(); }

  /**
   * @overload
   */
  const mfem::NewtonSolver* NonlinearSolver() const { return nonlin_solver_.get(); }

  /**
   * Returns the underlying linear solver object
   * @return A non-owning reference to the underlying linear solver
   */
  mfem::Solver* LinearSolver() { return lin_solver_.get(); }

  /**
   * @overload
   */
  const mfem::Solver* LinearSolver() const { return lin_solver_.get(); }

  /**
   * Returns the underlying linear solver object
   * @return A non-owning reference to the underlying linear solver
   */
  mfem::Solver* Preconditioner() { return preconditioner_.get(); }

  /**
   * @overload
   */
  const mfem::Solver* Preconditioner() const { return preconditioner_.get(); }

  /**
   * Input file parameters specific to this class
   **/
  static void DefineInputFileSchema(axom::inlet::Container& container);

private:
  /**
   * @brief The preconditioner (used for an iterative solver only)
   */
  std::unique_ptr<mfem::Solver> preconditioner_;

  /**
   * @brief The linear solver object, either custom, direct (SuperLU), or iterative
   */
  std::unique_ptr<mfem::Solver> lin_solver_;

  /**
   * @brief The optional nonlinear solver object
   */
  std::unique_ptr<mfem::NewtonSolver> nonlin_solver_;

  /**
   * @brief Whether the solver (linear solver) has been configured with the nonlinear solver
   * @note This is a workaround as some nonlinear solvers require SetOperator to be called
   * before SetSolver
   */
  bool nonlin_solver_set_solver_called_ = false;
};

/**
 * @brief A wrapper class for using the MFEM super LU solver with a HypreParMatrix
 */
class SuperLUSolver : public mfem::Solver {
public:
  /**
   * @brief Constructs a wrapper over an mfem::SuperLUSolver
   * @param[in] comm The MPI communicator used by the vectors and matrices in the solve
   * @param[in] print_level The verbosity level for the mfem::SuperLUSolver
   */
  SuperLUSolver(int print_level, MPI_Comm comm) : superlu_solver_(comm)
  {
    superlu_solver_.SetColumnPermutation(mfem::superlu::PARMETIS);
    if (print_level == 0) {
      superlu_solver_.SetPrintStatistics(false);
    }
  }

  /**
   * @brief Factor and solve the linear system y = Op^{-1} x using DSuperLU
   *
   * @param x The input RHS vector
   * @param y The output solution vector
   */
  void Mult(const mfem::Vector& x, mfem::Vector& y) const;

  /**
   * @brief Set the underlying matrix operator to use in the solution algorithm
   *
   * @param op The matrix operator to factorize with SuperLU
   * @pre This operator must be an assembled HypreParMatrix for compatibility with SuperLU
   */
  void SetOperator(const mfem::Operator& op);

private:
  /**
   * @brief The owner of the SuperLU matrix for the gradient, stored
   * as a member variable for lifetime purposes
   */
  mutable std::unique_ptr<mfem::SuperLURowLocMatrix> superlu_mat_;

  /**
   * @brief The underlying MFEM-based superLU solver. It requires a special
   * superLU matrix type which we store in this object. This enables compatibility
   * with HypreParMatrix when used as an input.
   */
  mfem::SuperLUSolver superlu_solver_;
};

/**
 * @brief Build an equation solver object using nonlinear and linear solver option structs.
 *
 * This constructed equation solver can then be passed directly into physics modules to
 * solve generic nonlinear systems of equations of the form F(x) = 0.
 *
 * @param nonlinear_opts The options to configure the nonlinear solution scheme
 * @param lin_opts The options to configure the underlying linear solution scheme to be used by the nonlinear solver
 * @param comm The MPI communicator for the supplied nonlinear operators and HypreParVectors
 * @return The constructed equation solver
 */
std::unique_ptr<EquationSolver> buildEquationSolver(NonlinearSolverOptions nonlinear_opts = {},
                                                    LinearSolverOptions lin_opts = {}, MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Build a nonlinear solver using the nonlinear option struct
 *
 * @param nonlinear_opts The options to configure the nonlinear solution scheme
 * @param comm The MPI communicator for the supplied nonlinear operators and HypreParVectors
 * @return The constructed nonlinear solver
 */
std::unique_ptr<mfem::NewtonSolver> buildNonlinearSolver(NonlinearSolverOptions nonlinear_opts = {},
                                                         MPI_Comm               comm           = MPI_COMM_WORLD);

/**
 * @brief Build the linear solver and its associated preconditioner given a linear options struct
 *
 * @param linear_opts The options to configure the linear solver and preconditioner
 * @param comm The MPI communicator for the supplied HypreParMatrix and HypreParVectors
 * @return A pair containing the constructed linear solver and preconditioner objects
 */
std::pair<std::unique_ptr<mfem::Solver>, std::unique_ptr<mfem::Solver>> buildLinearSolverAndPreconditioner(
    LinearSolverOptions linear_opts = {}, MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Build a preconditioner from the available options
 *
 * @param preconditioner The preconditioner type to be built
 * @param print_level The print level for the constructed preconditioner
 * @return A constructed preconditioner based on the input option
 */
std::unique_ptr<mfem::Solver> buildPreconditioner(Preconditioner preconditioner, int print_level = 0);

}  // namespace serac::mfem_ext

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by inlet
 */
template <>
struct FromInlet<serac::LinearSolverOptions> {
  /// @brief Returns created object from Inlet container
  serac::LinearSolverOptions operator()(const axom::inlet::Container& base);
};

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by inlet
 */
template <>
struct FromInlet<serac::NonlinearSolverOptions> {
  /// @brief Returns created object from Inlet container
  serac::NonlinearSolverOptions operator()(const axom::inlet::Container& base);
};

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by inlet
 */
template <>
struct FromInlet<serac::mfem_ext::EquationSolver> {
  /// @brief Returns created object from Inlet container
  serac::mfem_ext::EquationSolver operator()(const axom::inlet::Container& base);
};
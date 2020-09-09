// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file equation_solver.hpp
 *
 * @brief This file contains the declaration of an iterative solver wrapper
 */

#ifndef EQUATION_SOLVER
#define EQUATION_SOLVER

#include <memory>
#include <optional>
#include <variant>

#include "common/serac_types.hpp"
#include "mfem.hpp"

namespace serac {

/**
 * Wraps a (currently iterative) system solver and handles the configuration of linear
 * or nonlinear solvers.  This class solves a generic global system of (possibly) nonlinear algebraic equations.
 */
class EquationSolver : public mfem::Solver {
public:
  // TODO: Eliminate this once a dependency injection approach is used for the solvers
  EquationSolver() = default;
  /**
   * Constructs a new solver wrapper
   * @param[in] comm The MPI communicator object
   * @param[in] lin_params The parameters for the linear solver
   * @param[in] nonlin_params The optional parameters for the optional nonlinear solver
   * @see serac::LinearSolverParameters
   * @see serac::NonlinearSolverParameters
   */
  EquationSolver(MPI_Comm comm, const LinearSolverParameters& lin_params,
                 const std::optional<NonlinearSolverParameters>& nonlin_params = std::nullopt);

  /**
   * Sets a preconditioner for the underlying linear solver object
   * @param[in] prec The preconditioner, of which the object takes ownership
   * @note The preconditioner must be moved into the call
   * @code(.cpp)
   * solver.SetPreconditioner(std::move(prec));
   * @endcode
   */
  void SetPreconditioner(std::unique_ptr<mfem::Solver>&& prec)
  {
    prec_ = std::move(prec);
    iter_lin_solver_->SetPreconditioner(*prec_);
  }

  /**
   * Updates the solver with the provided operator
   * @param[in] op The operator (system matrix) to use, "A" in Ax = b
   * @note Implements mfem::Operator::SetOperator
   */
  void SetOperator(const mfem::Operator& op) override
  {
    solver().SetOperator(op);
    height = solver().Height();
    width  = solver().Width();
  }

  /**
   * Solves the system
   * @param[in] b RHS of the system of equations
   * @param[out] x Solution to the system of equations
   * @note Implements mfem::Operator::Mult
   */
  void Mult(const mfem::Vector& b, mfem::Vector& x) const override { solver().Mult(b, x); }

  /**
   * Returns the underlying solver object
   * @return The underlying nonlinear solver, if one was configured
   * when the object was constructed, otherwise, the underlying linear solver
   */
  mfem::IterativeSolver&       solver() { return (nonlin_solver_) ? **nonlin_solver_ : *iter_lin_solver_; }
  const mfem::IterativeSolver& solver() const { return (nonlin_solver_) ? **nonlin_solver_ : *iter_lin_solver_; }

  /**
   * Returns the underlying linear solver object, even if the class instance
   * has been configured as a nonlinear solver
   * @return The underlying linear solver
   */
  mfem::IterativeSolver&       linearSolver() { return *iter_lin_solver_; }
  const mfem::IterativeSolver& linearSolver() const { return *iter_lin_solver_; }

private:
  std::unique_ptr<mfem::IterativeSolver>                iter_lin_solver_;
  std::optional<std::unique_ptr<mfem::IterativeSolver>> nonlin_solver_;
  std::unique_ptr<mfem::Solver>                         prec_;
};

}  // namespace serac

#endif

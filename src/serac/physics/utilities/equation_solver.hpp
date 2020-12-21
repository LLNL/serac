// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file equation_solver.hpp
 *
 * @brief This file contains the declaration of an equation solver wrapper
 */

#ifndef EQUATION_SOLVER
#define EQUATION_SOLVER

#include <memory>
#include <optional>
#include <variant>

#include "mfem.hpp"
#include "serac/infrastructure/input.hpp"
#include "serac/physics/utilities/solver_config.hpp"

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
   * @param[in] lin_options The parameters for the linear solver
   * @param[in] nonlin_options The optional parameters for the optional nonlinear solver
   * @see serac::LinearSolverOptions
   * @see serac::NonlinearSolverOptions
   */
  EquationSolver(MPI_Comm comm, const LinearSolverOptions& lin_options,
                 const std::optional<NonlinearSolverOptions>& nonlin_options = std::nullopt);

  /**
   * Updates the solver with the provided operator
   * @param[in] op The operator (system matrix) to use, "A" in Ax = b
   * @note Implements mfem::Operator::SetOperator
   */
  void SetOperator(const mfem::Operator& op) override;

  /**
   * @brief An overload for "intercepting" HypreParMatrices
   * such that they can be converted to a SuperLURowLocMatrix
   * when running in SuperLU mode
   * @param[in] op The operator (system matrix) to use, "A" in Ax = b
   */
  void SetOperator(const mfem::HypreParMatrix& matrix);

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
  mfem::IterativeSolver&       nonlinearSolver() { return *nonlin_solver_; }
  const mfem::IterativeSolver& nonlinearSolver() const { return *nonlin_solver_; }

  /**
   * Returns the underlying linear solver object
   * @return A non-owning reference to the underlying linear solver
   */
  mfem::Solver& linearSolver()
  {
    return std::visit([](auto&& solver) -> mfem::Solver& { return *solver; }, lin_solver_);
  }
  const mfem::Solver& linearSolver() const
  {
    return std::visit([](auto&& solver) -> const mfem::Solver& { return *solver; }, lin_solver_);
  }

  /**
   * Input file parameters specific to this class
   **/
  static void defineInputFileSchema(axom::inlet::Table& table);

private:
  /**
   * @brief Builds an iterative solver given a set of linear solver parameters
   * @param[in] comm The MPI communicator object
   * @param[in] lin_options The parameters for the linear solver
   */
  std::unique_ptr<mfem::IterativeSolver> buildIterativeLinearSolver(MPI_Comm                      comm,
                                                                    const IterativeSolverOptions& lin_options);

  /**
   * @brief Builds an Newton-Raphson solver given a set of nonlinear solver parameters
   * @param[in] comm The MPI communicator object
   * @param[in] nonlin_options The parameters for the nonlinear solver
   */
  static std::unique_ptr<mfem::NewtonSolver> buildNewtonSolver(MPI_Comm                      comm,
                                                               const NonlinearSolverOptions& nonlin_options);

  /**
   * @brief A wrapper class for combining a nonlinear solver with a SuperLU direct solver
   */
  class SuperLUNonlinearOperatorWrapper : public mfem::Operator {
  public:
    /**
     * @brief Constructs a wrapper over an mfem::Operator
     * @param[in] oper The operator to wrap
     */
    SuperLUNonlinearOperatorWrapper(const mfem::Operator& oper) : oper_(oper)
    {
      height = oper_.Height();
      width  = oper_.Width();
    }
    /**
     * @brief Applies the operator
     * @param[in] b The input vector
     * @param[out] x The output vector
     * @note Implements mfem::Operator::Mult, forwards directly to underlying operator
     */
    void Mult(const mfem::Vector& b, mfem::Vector& x) const override { oper_.Mult(b, x); }

    /**
     * @brief Obtains the gradient of the underlying operator
     * as a SuperLU matrix
     * @param[in] x The point at which the gradient should be evaluated
     * @return A non-owning reference to an mfem::SuperLURowLocMatrix (upcasts
     * to match interface)
     * @note Implements mfem::Operator::GetGradient
     */
    mfem::Operator& GetGradient(const mfem::Vector& x) const override;

  private:
    /**
     * @brief The underlying operator
     */
    const mfem::Operator& oper_;

    /**
     * @brief The owner of the SuperLU matrix for the gradient, stored
     * as a member variable for lifetime purposes
     */
    mutable std::optional<mfem::SuperLURowLocMatrix> superlu_grad_mat_;
  };
  /**
   * @brief The preconditioner (used for an iterative solver only)
   */
  std::unique_ptr<mfem::Solver> prec_;

  /**
   * @brief The linear solver object, either custom, direct (SuperLU), or iterative
   */
  std::variant<std::unique_ptr<mfem::IterativeSolver>, std::unique_ptr<mfem::SuperLUSolver>, mfem::Solver*> lin_solver_;

  /**
   * @brief The optional nonlinear Newton-Raphson solver object
   */
  std::unique_ptr<mfem::NewtonSolver> nonlin_solver_;

  /**
   * @brief Whether the solver (linear solver) has been configured with the nonlinear solver
   * @note This is a workaround as some nonlinear solvers require SetOperator to be called
   * before SetSolver
   */
  bool nonlin_solver_set_solver_called_ = false;

  /**
   * @brief The operator (system matrix) used with a SuperLU solver
   */
  std::optional<mfem::SuperLURowLocMatrix> superlu_mat_;

  /**
   * @brief A wrapper class that allows a direct solver to be used underneath a Newton-Raphson solver
   */
  std::unique_ptr<SuperLUNonlinearOperatorWrapper> superlu_wrapper_;
};

/**
 * @brief A helper method intended to be called by physics modules to configure the AMG preconditioner for elasticity
 * problems
 * @param[in] init_options The user-provided solver parameters to possibly modify
 * @param[in] pfes The FiniteElementSpace to configure the preconditioner with
 * @note A full copy of the object is made, pending C++20 relaxation of "mutable"
 */
inline LinearSolverOptions augmentAMGForElasticity(const LinearSolverOptions&   init_options,
                                                   mfem::ParFiniteElementSpace& pfes)
{
  auto augmented_options = init_options;
  if (auto iter_options = std::get_if<IterativeSolverOptions>(&init_options)) {
    if (iter_options->prec) {
      if (std::holds_alternative<HypreBoomerAMGPrec>(iter_options->prec.value())) {
        // It's a copy, but at least it's on the stack
        std::get<HypreBoomerAMGPrec>(*std::get<IterativeSolverOptions>(augmented_options).prec).pfes = &pfes;
      }
    }
  }
  // NRVO will kick in here
  return augmented_options;
}

}  // namespace serac

// Prototype the specialization

template <>
struct FromInlet<serac::LinearSolverOptions> {
  serac::LinearSolverOptions operator()(const axom::inlet::Table& base);
};

template <>
struct FromInlet<serac::NonlinearSolverOptions> {
  serac::NonlinearSolverOptions operator()(const axom::inlet::Table& base);
};

template <>
struct FromInlet<serac::EquationSolver> {
  serac::EquationSolver operator()(const axom::inlet::Table& base);
};

#endif

// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "physics/utilities/equation_solver.hpp"

#include "infrastructure/logger.hpp"
#include "infrastructure/terminator.hpp"

namespace serac {

EquationSolver::EquationSolver(MPI_Comm comm, const LinearSolverParameters& lin_params,
                               const std::optional<NonlinearSolverParameters>& nonlin_params)
{
  if (lin_params.lin_solver == LinearSolver::SuperLU) {
    lin_solver_ = std::make_unique<mfem::SuperLUSolver>(comm);
    std::get<std::unique_ptr<mfem::SuperLUSolver>>(lin_solver_)->SetColumnPermutation(mfem::superlu::PARMETIS);
    if (lin_params.print_level == 0) {
      std::get<std::unique_ptr<mfem::SuperLUSolver>>(lin_solver_)->SetPrintStatistics(false);
    }
  } else {
    lin_solver_ = buildIterativeLinearSolver(comm, lin_params);
  }

  if (nonlin_params) {
    nonlin_solver_ = buildNewtonSolver(comm, *nonlin_params);
  }
}

std::unique_ptr<mfem::IterativeSolver> EquationSolver::buildIterativeLinearSolver(
    MPI_Comm comm, const LinearSolverParameters& lin_params)
{
  std::unique_ptr<mfem::IterativeSolver> iter_lin_solver;

  switch (lin_params.lin_solver) {
    case LinearSolver::CG:
      iter_lin_solver = std::make_unique<mfem::CGSolver>(comm);
      break;
    case LinearSolver::GMRES:
      iter_lin_solver = std::make_unique<mfem::GMRESSolver>(comm);
      break;
    case LinearSolver::MINRES:
      iter_lin_solver = std::make_unique<mfem::MINRESSolver>(comm);
      break;
    default:
      SLIC_ERROR("Linear solver type not recognized.");
      exitGracefully(true);
  }

  iter_lin_solver->SetRelTol(lin_params.rel_tol);
  iter_lin_solver->SetAbsTol(lin_params.abs_tol);
  iter_lin_solver->SetMaxIter(lin_params.max_iter);
  iter_lin_solver->SetPrintLevel(lin_params.print_level);

  return iter_lin_solver;
}

std::unique_ptr<mfem::NewtonSolver> EquationSolver::buildNewtonSolver(MPI_Comm                         comm,
                                                                      const NonlinearSolverParameters& nonlin_params)
{
  std::unique_ptr<mfem::NewtonSolver> newton_solver;

  if (nonlin_params.nonlin_solver == NonlinearSolver::BasicNewton) {
    newton_solver = std::make_unique<mfem::NewtonSolver>(comm);
  }
  // KINSOL
  else {
#ifdef MFEM_USE_SUNDIALS
    auto kinsol_strat = KIN_NONE;
    switch (nonlin_params.nonlin_solver) {
      case NonlinearSolver::KINBasic:
        break;  // Already the default
      case NonlinearSolver::KINLineSearch:
        kinsol_strat = KIN_LINESEARCH;
        break;
      case NonlinearSolver::KINFixedPoint:
        kinsol_strat = KIN_FP;
        break;
      case NonlinearSolver::KINPicard:
        kinsol_strat = KIN_PICARD;
        break;
      default:
        SLIC_WARNING("KINSOL option not recognized, defaulting to basic Newton.");
    }
    newton_solver = std::make_unique<mfem::KINSolver>(comm, kinsol_strat, true);
#else
    SLIC_ERROR("KINSOL was not enabled when MFEM was built");
#endif
  }

  newton_solver->SetRelTol(nonlin_params.rel_tol);
  newton_solver->SetAbsTol(nonlin_params.abs_tol);
  newton_solver->SetMaxIter(nonlin_params.max_iter);
  newton_solver->SetPrintLevel(nonlin_params.print_level);
  return newton_solver;
}

void EquationSolver::SetOperator(const mfem::Operator& op)
{
  if (nonlin_solver_) {
    if (std::holds_alternative<std::unique_ptr<mfem::SuperLUSolver>>(lin_solver_)) {
      superlu_wrapper_ = std::make_unique<SuperLUNonlinearOperatorWrapper>(op);
      nonlin_solver_->SetOperator(*superlu_wrapper_);
    } else {
      nonlin_solver_->SetOperator(op);
    }
    // Now that the nonlinear solver knows about the operator, we can set its linear solver
    if (!nonlin_solver_set_solver_called_) {
      nonlin_solver_->SetSolver(linearSolver());
      nonlin_solver_set_solver_called_ = true;
    }
  } else {
    std::visit([&op](auto&& solver) { solver->SetOperator(op); }, lin_solver_);
  }
  height = op.Height();
  width  = op.Width();
}

void EquationSolver::SetOperator(const mfem::HypreParMatrix& matrix)
{
  if (std::holds_alternative<std::unique_ptr<mfem::SuperLUSolver>>(lin_solver_)) {
    superlu_mat_ = matrix;
    SetOperator(*superlu_mat_);
  }
  // Otherwise just upcast and call as usual
  else {
    SetOperator(static_cast<const mfem::Operator&>(matrix));
  }
}

void EquationSolver::Mult(const mfem::Vector& b, mfem::Vector& x) const
{
  if (nonlin_solver_) {
    nonlin_solver_->Mult(b, x);
  } else {
    std::visit([&b, &x](auto&& solver) { solver->Mult(b, x); }, lin_solver_);
  }
}

void EquationSolver::SetPreconditioner(std::unique_ptr<mfem::Solver>&& prec)
{
  // If the linear solver is iterative, set the preconditioner
  if (std::holds_alternative<std::unique_ptr<mfem::IterativeSolver>>(lin_solver_)) {
    prec_ = std::move(prec);
    std::get<std::unique_ptr<mfem::IterativeSolver>>(lin_solver_)->SetPreconditioner(*prec_);
  }
}

mfem::Operator& EquationSolver::SuperLUNonlinearOperatorWrapper::GetGradient(const mfem::Vector& x) const
{
  mfem::Operator&       grad      = oper_.GetGradient(x);
  mfem::HypreParMatrix* matr_grad = dynamic_cast<mfem::HypreParMatrix*>(&grad);

  SLIC_ERROR_IF(matr_grad == nullptr, "Nonlinear operator gradient must be a HypreParMatrix");
  superlu_grad_mat_.emplace(*matr_grad);
  return *superlu_grad_mat_;
}

}  // namespace serac

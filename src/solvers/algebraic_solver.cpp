// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "solvers/algebraic_solver.hpp"

#include "common/logger.hpp"

namespace serac {

AlgebraicSolver::AlgebraicSolver(MPI_Comm comm, const LinearSolverParameters& lin_params,
                                 const std::optional<NonlinearSolverParameters>& nonlin_params)
{
  // Preconditioner configuration is too varied, maybe a PrecondParams is needed?
  // Maybe a redesign to better support custom preconditioners as well
  switch (lin_params.lin_solver) {
    case LinearSolver::CG:
      iter_lin_solver_ = std::make_unique<mfem::CGSolver>(comm);
      break;
    case LinearSolver::GMRES:
      iter_lin_solver_ = std::make_unique<mfem::GMRESSolver>(comm);
      break;
    case LinearSolver::MINRES:
      iter_lin_solver_ = std::make_unique<mfem::MINRESSolver>(comm);
      break;
    default:
      SLIC_ERROR("Linear solver type not recognized.");
      exitGracefully(true);
  }
  iter_lin_solver_->SetRelTol(lin_params.rel_tol);
  iter_lin_solver_->SetAbsTol(lin_params.abs_tol);
  iter_lin_solver_->SetMaxIter(lin_params.max_iter);
  iter_lin_solver_->SetPrintLevel(lin_params.print_level);

  if (nonlin_params) {
    auto newton_solver = std::make_unique<mfem::NewtonSolver>(comm);
    newton_solver->SetSolver(*iter_lin_solver_);
    newton_solver->SetRelTol(nonlin_params->rel_tol);
    newton_solver->SetAbsTol(nonlin_params->abs_tol);
    newton_solver->SetMaxIter(nonlin_params->max_iter);
    newton_solver->SetPrintLevel(nonlin_params->print_level);
    nonlin_solver_ = std::move(newton_solver);
  }
}

}  // namespace serac

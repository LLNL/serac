// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "solvers/equation_solver.hpp"

#include "common/common.hpp"

namespace serac {

EquationSolver::EquationSolver(MPI_Comm comm, const LinearSolverParameters& lin_params,
                               const std::optional<NonlinearSolverParameters>& nonlin_params)
{
  // Preconditioner configuration is too varied, maybe a PrecondParams is needed?
  // Maybe a redesign to better support custom preconditioners as well
  
  if (lin_params.lin_solver == LinearSolver::SuperLU) {
    lin_solver_ = std::make_unique<mfem::SuperLUSolver>(comm);
  } else {
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

    lin_solver_ = std::move(iter_lin_solver);
  } 

  if (nonlin_params) {
    auto newton_solver = std::make_unique<mfem::NewtonSolver>(comm);
    newton_solver->SetSolver(*lin_solver_);
    newton_solver->SetRelTol(nonlin_params->rel_tol);
    newton_solver->SetAbsTol(nonlin_params->abs_tol);
    newton_solver->SetMaxIter(nonlin_params->max_iter);
    newton_solver->SetPrintLevel(nonlin_params->print_level);
    nonlin_solver_ = std::move(newton_solver);
  }
}

void EquationSolver::SetOperator(const mfem::Operator& op)
{ 
  if (nonlinearSolver() != nullptr) {
    nonlinearSolver()->SetOperator(op); 
  } else {
    linearSolver()->SetOperator(op);
  }
}

void EquationSolver::Mult(const mfem::Vector& b, mfem::Vector& x) const
{ 
  if (nonlinearSolver() != nullptr) {
    nonlinearSolver()->Mult(b, x); 
  } else {
    linearSolver()->Mult(b, x); 
  }
}

void EquationSolver::SetPreconditioner(std::unique_ptr<mfem::Solver>&& prec)
{
  // If the linear solver is iterative, set the preconditioner
  auto iterative_solver = dynamic_cast<mfem::IterativeSolver*>(lin_solver_.get());
  if (iterative_solver != nullptr) {  
    prec_ = std::move(prec);
    iterative_solver->SetPreconditioner(*prec_);
  } else {
    SLIC_WARNING("Trying to set a preconditioner on a direct solver!");
  } 
}

}  // namespace serac

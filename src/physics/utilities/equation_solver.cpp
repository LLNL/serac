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
  // If it's an iterative solver, build it and set the preconditioner
  if (std::holds_alternative<IterativeSolverParameters>(lin_params)) {
    lin_solver_ = buildIterativeLinearSolver(comm, std::get<IterativeSolverParameters>(lin_params));
  }
  // If it's a custom solver, check that the mfem::Solver* is not null
  else if (std::holds_alternative<CustomSolverParameters>(lin_params)) {
    auto custom_solver_ptr = std::get<CustomSolverParameters>(lin_params).solver;
    SLIC_ERROR_IF(custom_solver_ptr == nullptr, "Custom solver pointer must be initialized.");
    lin_solver_ = custom_solver_ptr;
  }
  // If it's a direct solver (currently SuperLU only)
  else if (std::holds_alternative<DirectSolverParameters>(lin_params)) {
    auto direct_solver = std::make_unique<mfem::SuperLUSolver>(comm);
    direct_solver->SetColumnPermutation(mfem::superlu::PARMETIS);
    if (std::get<DirectSolverParameters>(lin_params).print_level == 0) {
      direct_solver->SetPrintStatistics(false);
    }
    lin_solver_ = std::move(direct_solver);
  }

  if (nonlin_params) {
    nonlin_solver_ = buildNewtonSolver(comm, *nonlin_params, linearSolver());
  }
}

std::unique_ptr<mfem::IterativeSolver> EquationSolver::buildIterativeLinearSolver(
    MPI_Comm comm, const IterativeSolverParameters& lin_params)
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

  // Handle the preconditioner - currently just BoomerAMG and HypreSmoother are supported
  if (std::holds_alternative<HypreBoomerAMGPrec>(lin_params.prec)) {
    auto par_fes = std::get<HypreBoomerAMGPrec>(lin_params.prec).pfes;
    SLIC_ERROR_IF(par_fes == nullptr, "FESpace is required to use the HypreBoomerAMG preconditioner.");
    SLIC_WARNING_IF(par_fes->GetOrdering() == mfem::Ordering::byNODES,
                    "Attempting to use BoomerAMG with nodal ordering.");
    auto prec_amg = std::make_unique<mfem::HypreBoomerAMG>();
    prec_amg->SetPrintLevel(lin_params.print_level);
    prec_amg->SetElasticityOptions(par_fes);
    prec_ = std::move(prec_amg);
  } else if (std::holds_alternative<HypreSmootherPrec>(lin_params.prec)) {
    auto relaxation_type = std::get<HypreSmootherPrec>(lin_params.prec).type;
    auto prec_smoother   = std::make_unique<mfem::HypreSmoother>();
    prec_smoother->SetType(relaxation_type);
    prec_smoother->SetPositiveDiagonal(true);
    prec_ = std::move(prec_smoother);
  }
  iter_lin_solver->SetPreconditioner(*prec_);
  return iter_lin_solver;
}

std::unique_ptr<mfem::NewtonSolver> EquationSolver::buildNewtonSolver(MPI_Comm                         comm,
                                                                      const NonlinearSolverParameters& nonlin_params,
                                                                      mfem::Solver&                    lin_solver)
{
  auto newton_solver = std::make_unique<mfem::NewtonSolver>(comm);
  newton_solver->SetSolver(lin_solver);
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

mfem::Operator& EquationSolver::SuperLUNonlinearOperatorWrapper::GetGradient(const mfem::Vector& x) const
{
  mfem::Operator&       grad      = oper_.GetGradient(x);
  mfem::HypreParMatrix* matr_grad = dynamic_cast<mfem::HypreParMatrix*>(&grad);

  SLIC_ERROR_IF(matr_grad == nullptr, "Nonlinear operator gradient must be a HypreParMatrix");
  superlu_grad_mat_.emplace(*matr_grad);
  return *superlu_grad_mat_;
}

void EquationSolver::defineInputFileSchema(std::shared_ptr<axom::inlet::SchemaCreator> schema_creator)
{
  auto nonlinear_table = schema_creator->addTable("nonlinear", "Newton Equation Solver Parameters")->required(true);
  nonlinear_table->addDouble("rel_tol", "Relative tolerance for the Newton solve.")->defaultValue(1.0e-2);
  nonlinear_table->addDouble("abs_tol", "Absolute tolerance for the Newton solve.")->defaultValue(1.0e-4);
  nonlinear_table->addInt("max_iter", "Maximum iterations for the Newton solve.")->defaultValue(500);
  nonlinear_table->addInt("print_level", "Nonlinear print level.")->defaultValue(0);
  nonlinear_table->addString("solver_type", "Not currently used.")->defaultValue("");

  auto linear_table = schema_creator->addTable("linear", "Linear Equation Solver Parameters")->required(true);
  linear_table->addDouble("rel_tol", "Relative tolerance for the linear solve.")->defaultValue(1.0e-6);
  linear_table->addDouble("abs_tol", "Absolute tolerance for the linear solve.")->defaultValue(1.0e-8);
  linear_table->addInt("max_iter", "Maximum iterations for the linear solve.")->defaultValue(5000);
  linear_table->addInt("print_level", "Linear print level.")->defaultValue(0);
  linear_table->addString("solver_type", "Solver type (gmres|minres).")->defaultValue("gmres");
}

}  // namespace serac

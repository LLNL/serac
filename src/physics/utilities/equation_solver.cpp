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
  if (auto iter_params = std::get_if<IterativeSolverParameters>(&lin_params)) {
    lin_solver_ = buildIterativeLinearSolver(comm, *iter_params);
  }
  // If it's a custom solver, check that the mfem::Solver* is not null
  else if (auto custom = std::get_if<CustomSolverParameters>(&lin_params)) {
    SLIC_ERROR_IF(custom->solver == nullptr, "Custom solver pointer must be initialized.");
    lin_solver_ = custom->solver;
  }
  // If it's a direct solver (currently SuperLU only)
  else if (auto direct_params = std::get_if<DirectSolverParameters>(&lin_params)) {
    auto direct_solver = std::make_unique<mfem::SuperLUSolver>(comm);
    direct_solver->SetColumnPermutation(mfem::superlu::PARMETIS);
    if (direct_params->print_level == 0) {
      direct_solver->SetPrintStatistics(false);
    }
    lin_solver_ = std::move(direct_solver);
  }

  if (nonlin_params) {
    nonlin_solver_ = buildNewtonSolver(comm, *nonlin_params);
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
  if (lin_params.prec) {
    const auto prec_ptr = &lin_params.prec.value();
    if (auto amg_params = std::get_if<HypreBoomerAMGPrec>(prec_ptr)) {
      auto prec_amg = std::make_unique<mfem::HypreBoomerAMG>();
      auto par_fes  = amg_params->pfes;
      if (par_fes != nullptr) {
        SLIC_WARNING_IF(par_fes->GetOrdering() == mfem::Ordering::byNODES,
                        "Attempting to use BoomerAMG with nodal ordering on an elasticity problem.");
        prec_amg->SetElasticityOptions(par_fes);
      }
      prec_amg->SetPrintLevel(lin_params.print_level);
      prec_ = std::move(prec_amg);
    } else if (auto smoother_params = std::get_if<HypreSmootherPrec>(prec_ptr)) {
      auto prec_smoother = std::make_unique<mfem::HypreSmoother>();
      prec_smoother->SetType(smoother_params->type);
      prec_smoother->SetPositiveDiagonal(true);
      prec_ = std::move(prec_smoother);
    } else if (std::holds_alternative<AMGXPrec>(*lin_params.prec)) {
#ifdef MFEM_USE_AMGX
      prec_ = std::make_unique<mfem::AmgXSolver>(comm, mfem::AmgXSolver::PRECONDITIONER, true);
#else
      SLIC_ERROR("AMGX was not enabled when MFEM was built");
#endif
    } else if (auto ilu_params = std::get_if<BlockILUPrec>(prec_ptr)) {
      prec_ = std::make_unique<mfem::BlockILU>(ilu_params->block_size);
    }
    iter_lin_solver->SetPreconditioner(*prec_);
  }
  return iter_lin_solver;
}

std::unique_ptr<mfem::NewtonSolver> EquationSolver::buildNewtonSolver(MPI_Comm                         comm,
                                                                      const NonlinearSolverParameters& nonlin_params)
{
  std::unique_ptr<mfem::NewtonSolver> newton_solver;

  if (nonlin_params.nonlin_solver == NonlinearSolver::MFEMNewton) {
    newton_solver = std::make_unique<mfem::NewtonSolver>(comm);
  }
  // KINSOL
  else {
#ifdef MFEM_USE_SUNDIALS
    auto kinsol_strat =
        (nonlin_params.nonlin_solver == NonlinearSolver::KINBacktrackingLineSearch) ? KIN_LINESEARCH : KIN_NONE;
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

mfem::Operator& EquationSolver::SuperLUNonlinearOperatorWrapper::GetGradient(const mfem::Vector& x) const
{
  mfem::Operator&       grad      = oper_.GetGradient(x);
  mfem::HypreParMatrix* matr_grad = dynamic_cast<mfem::HypreParMatrix*>(&grad);

  SLIC_ERROR_IF(matr_grad == nullptr, "Nonlinear operator gradient must be a HypreParMatrix");
  superlu_grad_mat_.emplace(*matr_grad);
  return *superlu_grad_mat_;
}

void EquationSolver::defineInputFileSchema(axom::inlet::Table& table)
{
  auto& linear_table = table.addTable("linear", "Linear Equation Solver Parameters").required();
  linear_table.addDouble("rel_tol", "Relative tolerance for the linear solve.").defaultValue(1.0e-6);
  linear_table.addDouble("abs_tol", "Absolute tolerance for the linear solve.").defaultValue(1.0e-8);
  linear_table.addInt("max_iter", "Maximum iterations for the linear solve.").defaultValue(5000);
  linear_table.addInt("print_level", "Linear print level.").defaultValue(0);
  linear_table.addString("solver_type", "Solver type (gmres|minres).").defaultValue("gmres");
  linear_table.addString("prec_type", "Preconditioner type (JacobiSmoother|L1JacobiSmoother|AMG|BlockILU).")
      .defaultValue("JacobiSmoother");

  // Only needed for nonlinear problems
  auto& nonlinear_table = table.addTable("nonlinear", "Newton Equation Solver Parameters").required(false);
  nonlinear_table.addDouble("rel_tol", "Relative tolerance for the Newton solve.").defaultValue(1.0e-2);
  nonlinear_table.addDouble("abs_tol", "Absolute tolerance for the Newton solve.").defaultValue(1.0e-4);
  nonlinear_table.addInt("max_iter", "Maximum iterations for the Newton solve.").defaultValue(500);
  nonlinear_table.addInt("print_level", "Nonlinear print level.").defaultValue(0);
  nonlinear_table.addString("solver_type", "Solver type (MFEMNewton|KINFullStep|KINLineSearch)")
      .defaultValue("MFEMNewton");
}

}  // namespace serac

using serac::EquationSolver;
using serac::IterativeSolverParameters;
using serac::NonlinearSolverParameters;

IterativeSolverParameters FromInlet<IterativeSolverParameters>::operator()(const axom::inlet::Table& base)
{
  IterativeSolverParameters params;
  params.rel_tol          = base["rel_tol"];
  params.abs_tol          = base["abs_tol"];
  params.max_iter         = base["max_iter"];
  params.print_level      = base["print_level"];
  std::string solver_type = base["solver_type"];
  if (solver_type == "gmres") {
    params.lin_solver = serac::LinearSolver::GMRES;
  } else if (solver_type == "minres") {
    params.lin_solver = serac::LinearSolver::MINRES;
  } else {
    std::string msg = fmt::format("Unknown Linear solver type given: {0}", solver_type);
    SLIC_ERROR(msg);
  }
  const std::string prec_type = base["prec_type"];
  if (prec_type == "JacobiSmoother") {
    params.prec = serac::HypreSmootherPrec{mfem::HypreSmoother::Jacobi};
  } else if (prec_type == "L1JacobiSmoother") {
    params.prec = serac::HypreSmootherPrec{mfem::HypreSmoother::l1Jacobi};
  } else if (prec_type == "HypreAMG") {
    params.prec = serac::HypreBoomerAMGPrec{};
  } else if (prec_type == "AMGX") {
    params.prec = serac::AMGXPrec{};
  } else if (prec_type == "BlockILU") {
    params.prec = serac::BlockILUPrec{};
  } else {
    std::string msg = fmt::format("Unknown preconditioner type given: {0}", prec_type);
    SLIC_ERROR(msg);
  }
  return params;
}

NonlinearSolverParameters FromInlet<NonlinearSolverParameters>::operator()(const axom::inlet::Table& base)
{
  NonlinearSolverParameters params;
  params.rel_tol                = base["rel_tol"];
  params.abs_tol                = base["abs_tol"];
  params.max_iter               = base["max_iter"];
  params.print_level            = base["print_level"];
  const std::string solver_type = base["solver_type"];
  if (solver_type == "MFEMNewton") {
    params.nonlin_solver = serac::NonlinearSolver::MFEMNewton;
  } else if (solver_type == "KINFullStep") {
    params.nonlin_solver = serac::NonlinearSolver::KINFullStep;
  } else if (solver_type == "KINLineSearch") {
    params.nonlin_solver = serac::NonlinearSolver::KINBacktrackingLineSearch;
  } else {
    SLIC_ERROR(fmt::format("Unknown nonlinear solver type given: {0}", solver_type));
  }
  return params;
}

EquationSolver FromInlet<EquationSolver>::operator()(const axom::inlet::Table& base)
{
  auto lin = base["linear"].get<IterativeSolverParameters>();
  if (base.hasTable("nonlinear")) {
    auto nonlin = base["nonlinear"].get<NonlinearSolverParameters>();
    return EquationSolver(MPI_COMM_WORLD, lin, nonlin);
  }
  return EquationSolver(MPI_COMM_WORLD, lin);
}

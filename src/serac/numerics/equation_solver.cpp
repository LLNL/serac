// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/numerics/equation_solver.hpp"

#include "serac/infrastructure/logger.hpp"
#include "serac/infrastructure/terminator.hpp"

namespace serac::mfem_ext {

EquationSolver::EquationSolver(std::unique_ptr<mfem::NewtonSolver> nonlinear_solver,
                               std::unique_ptr<mfem::Solver>       linear_solver,
                               std::unique_ptr<mfem::Solver>       preconditioner)
{
  nonlin_solver_  = std::move(nonlinear_solver);
  lin_solver_     = std::move(linear_solver);
  preconditioner_ = std::move(preconditioner);
}

void EquationSolver::SetOperator(const mfem::Operator& op)
{
  nonlin_solver_->SetOperator(op);

  auto* newton_solver = dynamic_cast<mfem::NewtonSolver*>(nonlin_solver_.get());

  if (newton_solver) {
    // Now that the nonlinear solver knows about the operator, we can set its linear solver
    if (!nonlin_solver_set_solver_called_) {
      if (LinearSolver()) {
        newton_solver->SetSolver(*LinearSolver());
      }
      nonlin_solver_set_solver_called_ = true;
    }
  }

  height = op.Height();
  width  = op.Width();
}

void EquationSolver::Mult(const mfem::Vector& b, mfem::Vector& x) const { nonlin_solver_->Mult(b, x); }

void SuperLUSolver::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
  SLIC_ERROR_ROOT_IF(!superlu_mat_, "Operator must be set prior to solving with SuperLU");

  // Use the underlying MFEM-based solver and SuperLU matrix type to solve the system
  superlu_solver_.Mult(x, y);
}

void SuperLUSolver::SetOperator(const mfem::Operator& op)
{
  const mfem::HypreParMatrix* matrix = dynamic_cast<const mfem::HypreParMatrix*>(&op);

  SLIC_ERROR_ROOT_IF(!matrix, "Matrix must be an assembled HypreParMatrix for use with SuperLU");
  superlu_mat_ = std::make_unique<mfem::SuperLURowLocMatrix>(*matrix);

  superlu_solver_.SetOperator(*superlu_mat_);
}

std::unique_ptr<mfem_ext::EquationSolver> buildEquationSolver(NonlinearSolverOptions nonlinear_opts,
                                                              LinearSolverOptions lin_opts, MPI_Comm comm)
{
  auto [linear_solver, preconditioner] = buildLinearSolverAndPreconditioner(lin_opts, comm);

  auto eq_solver = std::make_unique<mfem_ext::EquationSolver>(buildNonlinearSolver(nonlinear_opts, comm),
                                                              std::move(linear_solver), std::move(preconditioner));

  return eq_solver;
}

std::unique_ptr<mfem::NewtonSolver> buildNonlinearSolver(NonlinearSolverOptions nonlinear_opts, MPI_Comm comm)
{
  std::unique_ptr<mfem::NewtonSolver> nonlinear_solver;

  if (nonlinear_opts.nonlin_solver == NonlinearSolver::Newton) {
    nonlinear_solver = std::make_unique<mfem::NewtonSolver>(comm);
  } else if (nonlinear_opts.nonlin_solver == NonlinearSolver::LBFGS) {
    nonlinear_solver = std::make_unique<mfem::LBFGSSolver>(comm);
  }
  // KINSOL
  else {
#ifdef MFEM_USE_SUNDIALS

    int kinsol_strat = KIN_NONE;

    switch (nonlinear_opts.nonlin_solver) {
      case NonlinearSolver::KINFullStep:
        kinsol_strat = KIN_NONE;
        break;
      case NonlinearSolver::KINBacktrackingLineSearch:
        kinsol_strat = KIN_LINESEARCH;
        break;
      case NonlinearSolver::KINPicard:
        kinsol_strat = KIN_PICARD;
        break;
      default:
        kinsol_strat = KIN_NONE;
        SLIC_ERROR_ROOT("Unknown KINSOL nonlinear solver type given.");
    }
    auto kinsol_solver = std::make_unique<mfem::KINSolver>(comm, kinsol_strat, true);
    nonlinear_solver   = std::move(kinsol_solver);
#else
    SLIC_ERROR_ROOT("KINSOL was not enabled when MFEM was built");
#endif
  }

  nonlinear_solver->SetRelTol(nonlinear_opts.relative_tol);
  nonlinear_solver->SetAbsTol(nonlinear_opts.absolute_tol);
  nonlinear_solver->SetMaxIter(nonlinear_opts.max_iterations);
  nonlinear_solver->SetPrintLevel(nonlinear_opts.print_level);

  // Iterative mode indicates we do not zero out the initial guess during the
  // nonlinear solver call. This is required as we apply the essential boundary
  // conditions before the nonlinear solver is applied.
  nonlinear_solver->iterative_mode = true;

  return nonlinear_solver;
}

std::pair<std::unique_ptr<mfem::Solver>, std::unique_ptr<mfem::Solver>> buildLinearSolverAndPreconditioner(
    LinearSolverOptions linear_opts, MPI_Comm comm)
{
  if (linear_opts.linear_solver == LinearSolver::SuperLU) {
    auto lin_solver = std::make_unique<SuperLUSolver>(linear_opts.print_level, comm);
    return {std::move(lin_solver), nullptr};
  }

  std::unique_ptr<mfem::IterativeSolver> iter_lin_solver;

  switch (linear_opts.linear_solver) {
    case LinearSolver::CG:
      iter_lin_solver = std::make_unique<mfem::CGSolver>(comm);
      break;
    case LinearSolver::GMRES:
      iter_lin_solver = std::make_unique<mfem::GMRESSolver>(comm);
      break;
    default:
      SLIC_ERROR_ROOT("Linear solver type not recognized.");
      exitGracefully(true);
  }

  iter_lin_solver->SetRelTol(linear_opts.relative_tol);
  iter_lin_solver->SetAbsTol(linear_opts.absolute_tol);
  iter_lin_solver->SetMaxIter(linear_opts.max_iterations);
  iter_lin_solver->SetPrintLevel(linear_opts.print_level);

  auto preconditioner = buildPreconditioner(linear_opts.preconditioner, linear_opts.preconditioner_print_level, comm);

  if (preconditioner) {
    iter_lin_solver->SetPreconditioner(*preconditioner);
  }

  return {std::move(iter_lin_solver), std::move(preconditioner)};
}

#ifdef MFEM_USE_AMGX
std::unique_ptr<mfem::AmgXSolver> buildAMGX(const AMGXOptions& options, const MPI_Comm comm)
{
  auto          amgx = std::make_unique<mfem::AmgXSolver>();
  conduit::Node options_node;
  options_node["config_version"] = 2;
  auto& solver_options           = options_node["solver"];
  solver_options["solver"]       = "AMG";
  solver_options["presweeps"]    = 1;
  solver_options["postsweeps"]   = 2;
  solver_options["interpolator"] = "D2";
  solver_options["max_iters"]    = 2;
  solver_options["convergence"]  = "ABSOLUTE";
  solver_options["cycle"]        = "V";

  if (options.verbose) {
    options_node["solver/obtain_timings"]    = 1;
    options_node["solver/monitor_residual"]  = 1;
    options_node["solver/print_solve_stats"] = 1;
  }

  // TODO: Use magic_enum here when we can switch to GCC 9+
  // This is an immediately-invoked lambda so that the map
  // can be const without needed to initialize all the values
  // in the constructor
  static const auto solver_names = []() {
    std::unordered_map<AMGXSolver, std::string> names;
    names[AMGXSolver::AMG]             = "AMG";
    names[AMGXSolver::PCGF]            = "PCGF";
    names[AMGXSolver::CG]              = "CG";
    names[AMGXSolver::PCG]             = "PCG";
    names[AMGXSolver::PBICGSTAB]       = "PBICGSTAB";
    names[AMGXSolver::BICGSTAB]        = "BICGSTAB";
    names[AMGXSolver::FGMRES]          = "FGMRES";
    names[AMGXSolver::JACOBI_L1]       = "JACOBI_L1";
    names[AMGXSolver::GS]              = "GS";
    names[AMGXSolver::POLYNOMIAL]      = "POLYNOMIAL";
    names[AMGXSolver::KPZ_POLYNOMIAL]  = "KPZ_POLYNOMIAL";
    names[AMGXSolver::BLOCK_JACOBI]    = "BLOCK_JACOBI";
    names[AMGXSolver::MULTICOLOR_GS]   = "MULTICOLOR_GS";
    names[AMGXSolver::MULTICOLOR_DILU] = "MULTICOLOR_DILU";
    return names;
  }();

  options_node["solver/solver"]   = solver_names.at(options.solver);
  options_node["solver/smoother"] = solver_names.at(options.smoother);

  // Treat the string as the config (not a filename)
  amgx->ReadParameters(options_node.to_json(), mfem::AmgXSolver::INTERNAL);
  amgx->InitExclusiveGPU(comm);

  return amgx;
}
#endif

std::unique_ptr<mfem::Solver> buildPreconditioner(Preconditioner preconditioner, int print_level,
                                                  [[maybe_unused]] MPI_Comm comm)
{
  std::unique_ptr<mfem::Solver> preconditioner_ptr;

  // Handle the preconditioner - currently just BoomerAMG and HypreSmoother are supported
  if (preconditioner == Preconditioner::HypreAMG) {
    auto amg_preconditioner = std::make_unique<mfem::HypreBoomerAMG>();
    amg_preconditioner->SetPrintLevel(print_level);
    preconditioner_ptr = std::move(amg_preconditioner);
  } else if (preconditioner == Preconditioner::HypreJacobi) {
    auto jac_preconditioner = std::make_unique<mfem::HypreSmoother>();
    jac_preconditioner->SetType(mfem::HypreSmoother::Type::Jacobi);
    preconditioner_ptr = std::move(jac_preconditioner);
  } else if (preconditioner == Preconditioner::HypreL1Jacobi) {
    auto jacl1_preconditioner = std::make_unique<mfem::HypreSmoother>();
    jacl1_preconditioner->SetType(mfem::HypreSmoother::Type::l1Jacobi);
    preconditioner_ptr = std::move(jacl1_preconditioner);
  } else if (preconditioner == Preconditioner::HypreGaussSeidel) {
    auto gs_preconditioner = std::make_unique<mfem::HypreSmoother>();
    gs_preconditioner->SetType(mfem::HypreSmoother::Type::GS);
    preconditioner_ptr = std::move(gs_preconditioner);
  } else if (preconditioner == Preconditioner::AMGX) {
#ifdef MFEM_USE_AMGX
    preconditioner_ptr = buildAMGX(AMGXOptions{}, comm);
#else
    SLIC_ERROR_ROOT("AMGX requested in non-GPU build");
#endif
  } else {
    SLIC_ERROR_ROOT_IF(preconditioner != Preconditioner::None, "Unknown preconditioner type requested");
  }

  return preconditioner_ptr;
}

void EquationSolver::DefineInputFileSchema(axom::inlet::Container& container)
{
  auto& linear_container = container.addStruct("linear", "Linear Equation Solver Parameters");
  linear_container.required().registerVerifier([](const axom::inlet::Container& container_to_verify) {
    // Make sure that the provided options match the desired linear solver type
    const bool is_iterative = (container_to_verify["type"].get<std::string>() == "iterative") &&
                              container_to_verify.contains("iterative_options");
    const bool is_direct =
        (container_to_verify["type"].get<std::string>() == "direct") && container_to_verify.contains("direct_options");
    return is_iterative || is_direct;
  });

  // Enforce the solver type - must be iterative or direct
  linear_container.addString("type", "The type of solver parameters to use (iterative|direct)")
      .required()
      .validValues({"iterative", "direct"});

  auto& iterative_container = linear_container.addStruct("iterative_options", "Iterative solver parameters");
  iterative_container.addDouble("rel_tol", "Relative tolerance for the linear solve.").defaultValue(1.0e-6);
  iterative_container.addDouble("abs_tol", "Absolute tolerance for the linear solve.").defaultValue(1.0e-8);
  iterative_container.addInt("max_iter", "Maximum iterations for the linear solve.").defaultValue(5000);
  iterative_container.addInt("print_level", "Linear print level.").defaultValue(0);
  iterative_container.addString("solver_type", "Solver type (gmres|minres|cg).").defaultValue("gmres");
  iterative_container.addString("prec_type", "Preconditioner type (JacobiSmoother|L1JacobiSmoother|AMG|BlockILU).")
      .defaultValue("JacobiSmoother");

  auto& direct_container = linear_container.addStruct("direct_options", "Direct solver parameters");
  direct_container.addInt("print_level", "Linear print level.").defaultValue(0);

  // Only needed for nonlinear problems
  auto& nonlinear_container = container.addStruct("nonlinear", "Newton Equation Solver Parameters").required(false);
  nonlinear_container.addDouble("rel_tol", "Relative tolerance for the Newton solve.").defaultValue(1.0e-2);
  nonlinear_container.addDouble("abs_tol", "Absolute tolerance for the Newton solve.").defaultValue(1.0e-4);
  nonlinear_container.addInt("max_iter", "Maximum iterations for the Newton solve.").defaultValue(500);
  nonlinear_container.addInt("print_level", "Nonlinear print level.").defaultValue(0);
  nonlinear_container.addString("solver_type", "Solver type (Newton|KINFullStep|KINLineSearch)").defaultValue("Newton");
}

}  // namespace serac::mfem_ext

using serac::LinearSolverOptions;
using serac::NonlinearSolverOptions;
using serac::mfem_ext::EquationSolver;

serac::LinearSolverOptions FromInlet<serac::LinearSolverOptions>::operator()(const axom::inlet::Container& base)
{
  LinearSolverOptions options;
  std::string         type = base["type"];

  if (type == "direct") {
    options.linear_solver = serac::LinearSolver::SuperLU;
    options.print_level   = base["direct_options/print_level"];
    return options;
  }

  auto config             = base["iterative_options"];
  options.relative_tol    = config["rel_tol"];
  options.absolute_tol    = config["abs_tol"];
  options.max_iterations  = config["max_iter"];
  options.print_level     = config["print_level"];
  std::string solver_type = config["solver_type"];
  if (solver_type == "gmres") {
    options.linear_solver = serac::LinearSolver::GMRES;
  } else if (solver_type == "cg") {
    options.linear_solver = serac::LinearSolver::CG;
  } else {
    std::string msg = axom::fmt::format("Unknown Linear solver type given: '{0}'", solver_type);
    SLIC_ERROR_ROOT(msg);
  }
  const std::string prec_type = config["prec_type"];
  if (prec_type == "JacobiSmoother") {
    options.preconditioner = serac::Preconditioner::HypreJacobi;
  } else if (prec_type == "L1JacobiSmoother") {
    options.preconditioner = serac::Preconditioner::HypreL1Jacobi;
  } else if (prec_type == "HypreAMG") {
    options.preconditioner = serac::Preconditioner::HypreAMG;
#ifdef MFEM_USE_AMGX
  } else if (prec_type == "AMGX") {
    options.preconditioner = serac::Preconditioner::AMGX;
#endif
  } else if (prec_type == "GaussSeidel") {
    options.preconditioner = serac::Preconditioner::HypreGaussSeidel;
  } else {
    std::string msg = axom::fmt::format("Unknown preconditioner type given: '{0}'", prec_type);
    SLIC_ERROR_ROOT(msg);
  }

  return options;
}

serac::NonlinearSolverOptions FromInlet<serac::NonlinearSolverOptions>::operator()(const axom::inlet::Container& base)
{
  NonlinearSolverOptions options;
  options.relative_tol          = base["rel_tol"];
  options.absolute_tol          = base["abs_tol"];
  options.max_iterations        = base["max_iter"];
  options.print_level           = base["print_level"];
  const std::string solver_type = base["solver_type"];
  if (solver_type == "Newton") {
    options.nonlin_solver = serac::NonlinearSolver::Newton;
  } else if (solver_type == "KINFullStep") {
    options.nonlin_solver = serac::NonlinearSolver::KINFullStep;
  } else if (solver_type == "KINLineSearch") {
    options.nonlin_solver = serac::NonlinearSolver::KINBacktrackingLineSearch;
  } else if (solver_type == "KINPicard") {
    options.nonlin_solver = serac::NonlinearSolver::KINPicard;
  } else {
    SLIC_ERROR_ROOT(axom::fmt::format("Unknown nonlinear solver type given: '{0}'", solver_type));
  }
  return options;
}

serac::mfem_ext::EquationSolver FromInlet<serac::mfem_ext::EquationSolver>::operator()(
    const axom::inlet::Container& base)
{
  auto lin    = base["linear"].get<LinearSolverOptions>();
  auto nonlin = base["nonlinear"].get<NonlinearSolverOptions>();

  auto [linear_solver, preconditioner] = serac::mfem_ext::buildLinearSolverAndPreconditioner(lin, MPI_COMM_WORLD);

  serac::mfem_ext::EquationSolver eq_solver(serac::mfem_ext::buildNonlinearSolver(nonlin, MPI_COMM_WORLD),
                                            std::move(linear_solver), std::move(preconditioner));

  return eq_solver;
}
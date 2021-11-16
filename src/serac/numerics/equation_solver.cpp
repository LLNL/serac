// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/numerics/equation_solver.hpp"

#include "serac/infrastructure/logger.hpp"
#include "serac/infrastructure/terminator.hpp"

namespace serac::mfem_ext {

EquationSolver::EquationSolver(MPI_Comm comm, const LinearSolverOptions& lin_options,
                               const std::optional<NonlinearSolverOptions>& nonlin_options)
{
  // If it's an iterative solver, build it and set the preconditioner
  if (auto iter_options = std::get_if<IterativeSolverOptions>(&lin_options)) {
    lin_solver_ = BuildIterativeLinearSolver(comm, *iter_options);
  }
  // If it's a custom solver, check that the mfem::Solver* is not null
  else if (auto custom = std::get_if<CustomSolverOptions>(&lin_options)) {
    SLIC_ERROR_ROOT_IF(custom->solver == nullptr, "Custom solver pointer must be initialized.");
    lin_solver_ = custom->solver;
  }
  // If it's a direct solver (currently SuperLU only)
  else if (auto direct_options = std::get_if<DirectSolverOptions>(&lin_options)) {
    auto direct_solver = std::make_unique<mfem::SuperLUSolver>(comm);
    direct_solver->SetColumnPermutation(mfem::superlu::PARMETIS);
    if (direct_options->print_level == 0) {
      direct_solver->SetPrintStatistics(false);
    }
    lin_solver_ = std::move(direct_solver);
  }

  if (nonlin_options) {
    nonlin_solver_ = BuildNewtonSolver(comm, *nonlin_options);
  }
}

namespace detail {
#ifdef MFEM_USE_AMGX
std::unique_ptr<mfem::AmgXSolver> configureAMGX(const MPI_Comm comm, const AMGXPrec& options)
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

  // FIXME: magic_enum?
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
}  // namespace detail

std::unique_ptr<mfem::IterativeSolver> EquationSolver::BuildIterativeLinearSolver(
    MPI_Comm comm, const IterativeSolverOptions& lin_options)
{
  std::unique_ptr<mfem::IterativeSolver> iter_lin_solver;

  switch (lin_options.lin_solver) {
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
      SLIC_ERROR_ROOT("Linear solver type not recognized.");
      exitGracefully(true);
  }

  iter_lin_solver->SetRelTol(lin_options.rel_tol);
  iter_lin_solver->SetAbsTol(lin_options.abs_tol);
  iter_lin_solver->SetMaxIter(lin_options.max_iter);
  iter_lin_solver->SetPrintLevel(lin_options.print_level);

  // Handle the preconditioner - currently just BoomerAMG and HypreSmoother are supported
  if (lin_options.prec) {
    const auto prec_ptr = &lin_options.prec.value();
    if (auto amg_options = std::get_if<HypreBoomerAMGPrec>(prec_ptr)) {
      auto prec_amg = std::make_unique<mfem::HypreBoomerAMG>();
      auto par_fes  = amg_options->pfes;
      if (par_fes != nullptr) {
        SLIC_WARNING_ROOT_IF(par_fes->GetOrdering() == mfem::Ordering::byNODES,
                             "Attempting to use BoomerAMG with nodal ordering on an elasticity problem.");
        prec_amg->SetElasticityOptions(par_fes);
      }
      prec_amg->SetPrintLevel(lin_options.print_level);
      prec_ = std::move(prec_amg);
    } else if (auto smoother_options = std::get_if<HypreSmootherPrec>(prec_ptr)) {
      auto prec_smoother = std::make_unique<mfem::HypreSmoother>();
      prec_smoother->SetType(smoother_options->type);
      prec_smoother->SetPositiveDiagonal(true);
      prec_ = std::move(prec_smoother);
#ifdef MFEM_USE_AMGX
    } else if (auto amgx_options = std::get_if<AMGXPrec>(prec_ptr)) {
      prec_ = detail::configureAMGX(comm, *amgx_options);
#else
    } else if (std::get_if<AMGXPrec>(prec_ptr)) {
      SLIC_ERROR_ROOT("AMGX was not enabled when MFEM was built");
#endif
    } else if (auto ilu_options = std::get_if<BlockILUPrec>(prec_ptr)) {
      prec_ = std::make_unique<mfem::BlockILU>(ilu_options->block_size);
    }
    iter_lin_solver->SetPreconditioner(*prec_);
  }
  return iter_lin_solver;
}

std::unique_ptr<mfem::NewtonSolver> EquationSolver::BuildNewtonSolver(MPI_Comm                      comm,
                                                                      const NonlinearSolverOptions& nonlin_options)
{
  std::unique_ptr<mfem::NewtonSolver> newton_solver;

  if (nonlin_options.nonlin_solver == NonlinearSolver::MFEMNewton) {
    newton_solver = std::make_unique<mfem::NewtonSolver>(comm);
  }
  // KINSOL
  else {
#ifdef MFEM_USE_SUNDIALS
    auto kinsol_strat =
        (nonlin_options.nonlin_solver == NonlinearSolver::KINBacktrackingLineSearch) ? KIN_LINESEARCH : KIN_NONE;
    newton_solver = std::make_unique<mfem::KINSolver>(comm, kinsol_strat, true);
#else
    SLIC_ERROR_ROOT("KINSOL was not enabled when MFEM was built");
#endif
  }

  newton_solver->SetRelTol(nonlin_options.rel_tol);
  newton_solver->SetAbsTol(nonlin_options.abs_tol);
  newton_solver->SetMaxIter(nonlin_options.max_iter);
  newton_solver->SetPrintLevel(nonlin_options.print_level);
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
      nonlin_solver_->SetSolver(LinearSolver());
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

  SLIC_ERROR_ROOT_IF(matr_grad == nullptr, "Nonlinear operator gradient must be a HypreParMatrix");
  superlu_grad_mat_.emplace(*matr_grad);
  return *superlu_grad_mat_;
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
  nonlinear_container.addString("solver_type", "Solver type (MFEMNewton|KINFullStep|KINLineSearch)")
      .defaultValue("MFEMNewton");
}

}  // namespace serac::mfem_ext

using serac::LinearSolverOptions;
using serac::NonlinearSolverOptions;
using serac::mfem_ext::EquationSolver;

serac::LinearSolverOptions FromInlet<serac::LinearSolverOptions>::operator()(const axom::inlet::Container& base)
{
  LinearSolverOptions options;
  std::string         type = base["type"];
  if (type == "iterative") {
    serac::IterativeSolverOptions iter_options;
    auto                          config = base["iterative_options"];
    iter_options.rel_tol                 = config["rel_tol"];
    iter_options.abs_tol                 = config["abs_tol"];
    iter_options.max_iter                = config["max_iter"];
    iter_options.print_level             = config["print_level"];
    std::string solver_type              = config["solver_type"];
    if (solver_type == "gmres") {
      iter_options.lin_solver = serac::LinearSolver::GMRES;
    } else if (solver_type == "minres") {
      iter_options.lin_solver = serac::LinearSolver::MINRES;
    } else if (solver_type == "cg") {
      iter_options.lin_solver = serac::LinearSolver::CG;
    } else {
      std::string msg = axom::fmt::format("Unknown Linear solver type given: {0}", solver_type);
      SLIC_ERROR_ROOT(msg);
    }
    const std::string prec_type = config["prec_type"];
    if (prec_type == "JacobiSmoother") {
      iter_options.prec = serac::HypreSmootherPrec{mfem::HypreSmoother::Jacobi};
    } else if (prec_type == "L1JacobiSmoother") {
      iter_options.prec = serac::HypreSmootherPrec{mfem::HypreSmoother::l1Jacobi};
    } else if (prec_type == "HypreAMG") {
      iter_options.prec = serac::HypreBoomerAMGPrec{};
    } else if (prec_type == "AMGX") {
      iter_options.prec = serac::AMGXPrec{};
    } else if (prec_type == "L1JacobiAMGX") {
      iter_options.prec = serac::AMGXPrec{.smoother = serac::AMGXSolver::JACOBI_L1};
    } else if (prec_type == "BlockILU") {
      iter_options.prec = serac::BlockILUPrec{};
    } else {
      std::string msg = axom::fmt::format("Unknown preconditioner type given: {0}", prec_type);
      SLIC_ERROR_ROOT(msg);
    }
    options = iter_options;
  } else if (type == "direct") {
    serac::DirectSolverOptions direct_options;
    direct_options.print_level = base["direct_options/print_level"];
    options                    = direct_options;
  }
  return options;
}

serac::NonlinearSolverOptions FromInlet<serac::NonlinearSolverOptions>::operator()(const axom::inlet::Container& base)
{
  NonlinearSolverOptions options;
  options.rel_tol               = base["rel_tol"];
  options.abs_tol               = base["abs_tol"];
  options.max_iter              = base["max_iter"];
  options.print_level           = base["print_level"];
  const std::string solver_type = base["solver_type"];
  if (solver_type == "MFEMNewton") {
    options.nonlin_solver = serac::NonlinearSolver::MFEMNewton;
  } else if (solver_type == "KINFullStep") {
    options.nonlin_solver = serac::NonlinearSolver::KINFullStep;
  } else if (solver_type == "KINLineSearch") {
    options.nonlin_solver = serac::NonlinearSolver::KINBacktrackingLineSearch;
  } else {
    SLIC_ERROR_ROOT(axom::fmt::format("Unknown nonlinear solver type given: {0}", solver_type));
  }
  return options;
}

serac::mfem_ext::EquationSolver FromInlet<serac::mfem_ext::EquationSolver>::operator()(
    const axom::inlet::Container& base)
{
  auto lin = base["linear"].get<LinearSolverOptions>();
  if (base.contains("nonlinear")) {
    auto nonlin = base["nonlinear"].get<NonlinearSolverOptions>();
    return EquationSolver(MPI_COMM_WORLD, lin, nonlin);
  }
  return EquationSolver(MPI_COMM_WORLD, lin);
}

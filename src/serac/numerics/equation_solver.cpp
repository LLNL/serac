// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/numerics/equation_solver.hpp"
#include <iomanip>
#include <sstream>
#include <ios>
#include <iostream>

#include "serac/infrastructure/logger.hpp"
#include "serac/infrastructure/terminator.hpp"
#include "serac/serac_config.hpp"

namespace serac {

class NewtonSolver : public mfem::NewtonSolver {
protected:

  mutable mfem::Vector x0;

public:
  NewtonSolver() {}

#ifdef MFEM_USE_MPI
  NewtonSolver(MPI_Comm comm_) : mfem::NewtonSolver(comm_) {}
#endif

  void Mult(const mfem::Vector&, mfem::Vector& x) const
  {
    MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
    MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");

    using real_t = mfem::real_t;

    real_t norm, norm_goal;
    oper->Mult(x, r);

    norm = initial_norm = Norm(r);
    if (print_options.first_and_last && !print_options.iterations) {
      mfem::out << "Newton iteration " << std::setw(2) << 0 << " : ||r|| = " << norm << "...\n";
    }

    norm_goal            = std::max(rel_tol * initial_norm, abs_tol);
    prec->iterative_mode = false;

    int it = 0;
    for (; true; it++) {
      MFEM_ASSERT(IsFinite(norm), "norm = " << norm);
      if (print_options.iterations) {
        mfem::out << "Newton iteration " << std::setw(2) << it << " : ||r|| = " << norm;
        if (it > 0) {
          mfem::out << ", ||r||/||r_0|| = " << norm / initial_norm;
        }
        mfem::out << '\n';
      }

      if (norm <= norm_goal) {
        converged = true;
        break;
      }

      if (it >= max_iter) {
        converged = false;
        break;
      }

      real_t norm_nm1 = norm;

      grad = &oper->GetGradient(x);
      prec->SetOperator(*grad);

      prec->Mult(r, c);  // c = [DF(x_i)]^{-1} [F(x_i)-b]
  
      // there must be a better way to do this?
      x0.SetSize(x.Size());
      x0 = 0.0;
      x0.Add(1.0, x);

      real_t c_scale = 1.0;
      add(x0, -c_scale, c, x);
      oper->Mult(x, r);
      norm = Norm(r);

      static constexpr int max_ls_iters = 20;
      static constexpr real_t reduction = 0.5;

      auto is_converged = [=](real_t currentNorm, real_t) {
        return currentNorm < norm_nm1;
      };

      // back-track linesearch
      int ls_iter     = 0;
      int ls_iter_sum = 0.0;
      for (; !is_converged(norm, c_scale) && ls_iter < max_ls_iters; ++ls_iter, ++ls_iter_sum) {
        c_scale *= reduction;
        add(x0, -c_scale, c, x);
        oper->Mult(x, r);
        norm = Norm(r);
      }

      // try the opposite direction and linesearch back from there
      if (ls_iter==max_ls_iters && !is_converged(norm, c_scale)) {

        // std::cout << std::setprecision(15) << "tmp norm = " << norm << " " << norm_nm1 << " " << ls_iter << " " << c_scale << " " << std::endl;

        c_scale = 1.0;
        add(x0, c_scale, c, x);
        oper->Mult(x, r);
        norm = Norm(r);

        ls_iter = 0;
        for (; !is_converged(norm, c_scale) && ls_iter < max_ls_iters; ++ls_iter, ++ls_iter_sum) {
          c_scale *= reduction;
          add(x0, c_scale, c, x);
          oper->Mult(x, r);
          norm = Norm(r);
        }

        // std::cout << std::setprecision(15) << "int norm = " << norm << " " << norm_nm1 << " " << ls_iter_sum << " " << c_scale << " " << std::endl;

        // ok, the opposite direction was also terrible, lets go back, cut in half 1 last time and accept it
        if (ls_iter==max_ls_iters && !is_converged(norm, c_scale)) {
          ++ls_iter_sum;
          add(x0, -reduction * c_scale, c, x);
          oper->Mult(x, r);
          norm = Norm(r);
        }
      }
      // std::cout << std::setprecision(15) << "new norm = " << norm << " " << norm_nm1 << " " << ls_iter_sum << " " << c_scale << " " << std::endl;
    }

    final_iter = it;
    final_norm = norm;

    if (print_options.summary || (!converged && print_options.warnings) || print_options.first_and_last) {
      mfem::out << "Newton: Number of iterations: " << final_iter << '\n' << "   ||r|| = " << final_norm << '\n';
    }
    if (!converged && (print_options.summary || print_options.warnings)) {
      mfem::out << "Newton: No convergence!\n";
    }
  }
};

EquationSolver::EquationSolver(NonlinearSolverOptions nonlinear_opts, LinearSolverOptions lin_opts, MPI_Comm comm)
{
  auto [lin_solver, preconditioner] = buildLinearSolverAndPreconditioner(lin_opts, comm);

  lin_solver_     = std::move(lin_solver);
  preconditioner_ = std::move(preconditioner);
  nonlin_solver_  = buildNonlinearSolver(nonlinear_opts, comm);
}

EquationSolver::EquationSolver(std::unique_ptr<mfem::NewtonSolver> nonlinear_solver,
                               std::unique_ptr<mfem::Solver>       linear_solver,
                               std::unique_ptr<mfem::Solver>       preconditioner)
{
  SLIC_ERROR_ROOT_IF(!nonlinear_solver, "Nonlinear solvers must be given to construct an EquationSolver");
  SLIC_ERROR_ROOT_IF(!linear_solver, "Linear solvers must be given to construct an EquationSolver");

  nonlin_solver_  = std::move(nonlinear_solver);
  lin_solver_     = std::move(linear_solver);
  preconditioner_ = std::move(preconditioner);
}

void EquationSolver::setOperator(const mfem::Operator& op)
{
  nonlin_solver_->SetOperator(op);

  // Now that the nonlinear solver knows about the operator, we can set its linear solver
  if (!nonlin_solver_set_solver_called_) {
    nonlin_solver_->SetSolver(linearSolver());
    nonlin_solver_set_solver_called_ = true;
  }
}

void EquationSolver::solve(mfem::Vector& x) const
{
  mfem::Vector zero(x);
  zero = 0.0;
  // KINSOL does not handle non-zero RHS, so we enforce that the RHS
  // of the nonlinear system is zero
  nonlin_solver_->Mult(zero, x);
}

void SuperLUSolver::Mult(const mfem::Vector& input, mfem::Vector& output) const
{
  SLIC_ERROR_ROOT_IF(!superlu_mat_, "Operator must be set prior to solving with SuperLU");

  // Use the underlying MFEM-based solver and SuperLU matrix type to solve the system
  superlu_solver_.Mult(input, output);
}

/**
 * @brief Function for building a monolithic parallel Hypre matrix from a block system of smaller Hypre matrices
 *
 * @param block_operator The block system of HypreParMatrices
 * @return The assembled monolithic HypreParMatrix
 *
 * @pre @a block_operator must have assembled HypreParMatrices for its sub-blocks
 */
std::unique_ptr<mfem::HypreParMatrix> buildMonolithicMatrix(const mfem::BlockOperator& block_operator)
{
  int row_blocks = block_operator.NumRowBlocks();
  int col_blocks = block_operator.NumColBlocks();

  SLIC_ERROR_ROOT_IF(row_blocks != col_blocks, "Attempted to use a direct solver on a non-square block system.");

  mfem::Array2D<const mfem::HypreParMatrix*> hypre_blocks(row_blocks, col_blocks);

  for (int i = 0; i < row_blocks; ++i) {
    for (int j = 0; j < col_blocks; ++j) {
      // checks for presence of empty (null) blocks, which happen fairly common in multirank contact
      if (!block_operator.IsZeroBlock(i, j)) {
        auto* hypre_block = dynamic_cast<const mfem::HypreParMatrix*>(&block_operator.GetBlock(i, j));
        SLIC_ERROR_ROOT_IF(!hypre_block,
                           "Trying to use SuperLU on a block operator that does not contain HypreParMatrix blocks.");

        hypre_blocks(i, j) = hypre_block;
      } else {
        hypre_blocks(i, j) = nullptr;
      }
    }
  }

  // Note that MFEM passes ownership of this matrix to the caller
  return std::unique_ptr<mfem::HypreParMatrix>(mfem::HypreParMatrixFromBlocks(hypre_blocks));
}

void SuperLUSolver::SetOperator(const mfem::Operator& op)
{
  // Check if this is a block operator
  auto* block_operator = dynamic_cast<const mfem::BlockOperator*>(&op);

  // If it is, make a monolithic system from the underlying blocks
  if (block_operator) {
    auto monolithic_mat = buildMonolithicMatrix(*block_operator);

    superlu_mat_ = std::make_unique<mfem::SuperLURowLocMatrix>(*monolithic_mat);
  } else {
    // If this is not a block system, check that the input operator is a HypreParMatrix as expected
    auto* matrix = dynamic_cast<const mfem::HypreParMatrix*>(&op);

    SLIC_ERROR_ROOT_IF(!matrix, "Matrix must be an assembled HypreParMatrix for use with SuperLU");

    superlu_mat_ = std::make_unique<mfem::SuperLURowLocMatrix>(*matrix);
  }

  superlu_solver_.SetOperator(*superlu_mat_);
}

#ifdef MFEM_USE_STRUMPACK

void StrumpackSolver::Mult(const mfem::Vector& input, mfem::Vector& output) const
{
  SLIC_ERROR_ROOT_IF(!strumpack_mat_, "Operator must be set prior to solving with Strumpack");

  // Use the underlying MFEM-based solver and Strumpack matrix type to solve the system
  strumpack_solver_.Mult(input, output);
}

void StrumpackSolver::SetOperator(const mfem::Operator& op)
{
  // Check if this is a block operator
  auto* block_operator = dynamic_cast<const mfem::BlockOperator*>(&op);

  // If it is, make a monolithic system from the underlying blocks
  if (block_operator) {
    auto monolithic_mat = buildMonolithicMatrix(*block_operator);

    strumpack_mat_ = std::make_unique<mfem::STRUMPACKRowLocMatrix>(*monolithic_mat);
  } else {
    // If this is not a block system, check that the input operator is a HypreParMatrix as expected
    auto* matrix = dynamic_cast<const mfem::HypreParMatrix*>(&op);

    SLIC_ERROR_ROOT_IF(!matrix, "Matrix must be an assembled HypreParMatrix for use with Strumpack");

    strumpack_mat_ = std::make_unique<mfem::STRUMPACKRowLocMatrix>(*matrix);
  }

  strumpack_solver_.SetOperator(*strumpack_mat_);
}

#endif

std::unique_ptr<mfem::NewtonSolver> buildNonlinearSolver(NonlinearSolverOptions nonlinear_opts, MPI_Comm comm)
{
  std::unique_ptr<mfem::NewtonSolver> nonlinear_solver;

  if (nonlinear_opts.nonlin_solver == NonlinearSolver::Newton) {
    nonlinear_solver = std::make_unique<mfem::NewtonSolver>(comm);
  } else if (nonlinear_opts.nonlin_solver == NonlinearSolver::LBFGS) {
    nonlinear_solver = std::make_unique<mfem::LBFGSSolver>(comm);
  } else if (nonlinear_opts.nonlin_solver == NonlinearSolver::NewtonLineSearch) {
    nonlinear_solver = std::make_unique<NewtonSolver>(comm);
  }
  // KINSOL
  else {
#ifdef SERAC_USE_SUNDIALS

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

#ifdef MFEM_USE_STRUMPACK

  if (linear_opts.linear_solver == LinearSolver::Strumpack) {
    auto lin_solver = std::make_unique<StrumpackSolver>(linear_opts.print_level, comm);
    return {std::move(lin_solver), nullptr};
  }

#endif

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
  std::unique_ptr<mfem::Solver> preconditioner_solver;

  // Handle the preconditioner - currently just BoomerAMG and HypreSmoother are supported
  if (preconditioner == Preconditioner::HypreAMG) {
    auto amg_preconditioner = std::make_unique<mfem::HypreBoomerAMG>();
    amg_preconditioner->SetPrintLevel(print_level);
    preconditioner_solver = std::move(amg_preconditioner);
  } else if (preconditioner == Preconditioner::HypreJacobi) {
    auto jac_preconditioner = std::make_unique<mfem::HypreSmoother>();
    jac_preconditioner->SetType(mfem::HypreSmoother::Type::Jacobi);
    preconditioner_solver = std::move(jac_preconditioner);
  } else if (preconditioner == Preconditioner::HypreL1Jacobi) {
    auto jacl1_preconditioner = std::make_unique<mfem::HypreSmoother>();
    jacl1_preconditioner->SetType(mfem::HypreSmoother::Type::l1Jacobi);
    preconditioner_solver = std::move(jacl1_preconditioner);
  } else if (preconditioner == Preconditioner::HypreGaussSeidel) {
    auto gs_preconditioner = std::make_unique<mfem::HypreSmoother>();
    gs_preconditioner->SetType(mfem::HypreSmoother::Type::GS);
    preconditioner_solver = std::move(gs_preconditioner);
  } else if (preconditioner == Preconditioner::HypreILU) {
    auto ilu_preconditioner = std::make_unique<mfem::HypreILU>();
    ilu_preconditioner->SetLevelOfFill(1);
    ilu_preconditioner->SetPrintLevel(print_level);
    preconditioner_solver = std::move(ilu_preconditioner);
  } else if (preconditioner == Preconditioner::AMGX) {
#ifdef MFEM_USE_AMGX
    preconditioner_solver = buildAMGX(AMGXOptions{}, comm);
#else
    SLIC_ERROR_ROOT("AMGX requested in non-GPU build");
#endif
  } else {
    SLIC_ERROR_ROOT_IF(preconditioner != Preconditioner::None, "Unknown preconditioner type requested");
  }

  return preconditioner_solver;
}

void EquationSolver::defineInputFileSchema(axom::inlet::Container& container)
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
  iterative_container.addString("prec_type", "Preconditioner type (JacobiSmoother|L1JacobiSmoother|AMG|ILU).")
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

}  // namespace serac

using serac::EquationSolver;
using serac::LinearSolverOptions;
using serac::NonlinearSolverOptions;

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
  } else if (prec_type == "ILU") {
    options.preconditioner = serac::Preconditioner::HypreILU;
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

serac::EquationSolver FromInlet<serac::EquationSolver>::operator()(const axom::inlet::Container& base)
{
  auto lin    = base["linear"].get<LinearSolverOptions>();
  auto nonlin = base["nonlinear"].get<NonlinearSolverOptions>();

  auto [linear_solver, preconditioner] = serac::buildLinearSolverAndPreconditioner(lin, MPI_COMM_WORLD);

  serac::EquationSolver eq_solver(serac::buildNonlinearSolver(nonlin, MPI_COMM_WORLD), std::move(linear_solver),
                                  std::move(preconditioner));

  return eq_solver;
}

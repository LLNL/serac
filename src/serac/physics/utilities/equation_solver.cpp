// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/utilities/equation_solver.hpp"

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
    SLIC_ERROR_IF(custom->solver == nullptr, "Custom solver pointer must be initialized.");
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
using JSONLiteral  = std::variant<std::string, double, int>;
using JSONLiterals = std::unordered_map<std::string, JSONLiteral>;

/**
 * @brief A structure for storing simple JSON data
 *
 * Used to produce the configuration for AMGX
 */
class JSONTable {
public:
  /**
   * @brief Creates a new table
   * @param[in] depth The depth of the created table
   */
  JSONTable(const int depth = 0) : depth_(depth) {}
  /**
   * @brief Returns the subtable at the given index, creating a new table if necessary
   * @param[in] idx The string key of the subtable (object)
   */
  JSONTable& operator[](const std::string& idx)
  {
    if (tables_.count(idx) == 0) {
      tables_[idx] = std::make_unique<JSONTable>(depth_ + 1);
    }
    return *(tables_[idx]);
  }

  /**
   * @brief Returns the literal at the given index, creating a new one if it doesn't exist
   * @param[in] idx The string key of the subtable (object)
   */
  JSONLiteral& literal(const std::string& idx) { return literals_[idx]; }

  /**
   * @brief Adds (appends) a table to the calling table
   * @param[in] other The table to add
   */
  void add(const JSONTable& other)
  {
    literals_.insert(other.literals_.begin(), other.literals_.end());
    for (const auto& [key, subtable] : other.tables_) {
      operator[](key).add(*subtable);
    }
  }

  /**
   * @brief Inserts the JSON representation to a stream
   * @param[inout] out The stream to insert into
   * @param[in] table The table to be inserted (printed)
   */
  friend std::ostream& operator<<(std::ostream& out, const JSONTable& table);

private:
  /**
   * @brief Literal (primitive) members of the table
   */
  JSONLiterals literals_;
  /**
   * @brief Subtable (object) members of the table
   */
  std::unordered_map<std::string, std::unique_ptr<JSONTable>> tables_;
  /**
   * @brief Current depth of the table relative to the root
   */
  const int depth_;
};

std::ostream& operator<<(std::ostream& out, const JSONTable& table)
{
  out << "{";
  std::string indent(static_cast<std::size_t>(table.depth_) * 2, ' ');  // Double-space indenting
  char        sep = ' ';  // Start with empty separator to avoid trailing comma
  for (const auto& [key, val] : table.literals_) {
    out << sep << "\n" << indent << "\"" << key << "\": ";
    // Strings need to be wrapped in escaped quotes
    if (auto str_ptr = std::get_if<std::string>(&val)) {
      out << "\"" << *str_ptr << "\"";
    } else {
      std::visit([&out](const auto contained_val) { out << contained_val; }, val);
    }
    sep = ',';
  }

  // Recursively insert subtables into the stream
  for (const auto& [key, subtable] : table.tables_) {
    out << sep << "\n" << indent << "\"" << key << "\": ";
    out << *subtable;
    sep = ',';
  }
  out << "\n" << indent << "}";
  return out;
}

std::unique_ptr<mfem::AmgXSolver> configureAMGX(const MPI_Comm comm, const AMGXPrec& options)
{
  auto              amgx                 = std::make_unique<mfem::AmgXSolver>();
  static const auto default_prec_options = []() {
    JSONTable solver_table;
    solver_table.literal("solver")       = "AMG";
    solver_table.literal("presweeps")    = 1;
    solver_table.literal("postsweeps")   = 2;
    solver_table.literal("interpolator") = "D2";
    solver_table.literal("max_iters")    = 2;
    solver_table.literal("convergence")  = "ABSOLUTE";
    solver_table.literal("cycle")        = "V";

    JSONTable top_level;
    top_level.literal("config_version") = 2;
    top_level["solver"].add(solver_table);
    return top_level;
  }();

  JSONTable options_table;
  options_table.add(default_prec_options);
  if (options.verbose) {
    static const auto default_verbose_options = []() {
      JSONTable result;
      result.literal("obtain_timings")    = 1;
      result.literal("monitor_residual")  = 1;
      result.literal("print_solve_stats") = 1;
      result.literal("print_solve_stats") = 1;
      return result;
    }();
    options_table["solver"].add(default_verbose_options);
  }

  // FIXME: magic_enum?
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

  options_table["solver"].literal("solver")   = solver_names.at(options.solver);
  options_table["solver"].literal("smoother") = solver_names.at(options.smoother);

  std::ostringstream oss;
  oss << options_table;
  // Treat the string as the config (not a filename)
  amgx->ReadParameters(oss.str(), mfem::AmgXSolver::INTERNAL);
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
      SLIC_ERROR("Linear solver type not recognized.");
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
        SLIC_WARNING_IF(par_fes->GetOrdering() == mfem::Ordering::byNODES,
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
      SLIC_ERROR("AMGX was not enabled when MFEM was built");
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
    SLIC_ERROR("KINSOL was not enabled when MFEM was built");
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

  SLIC_ERROR_IF(matr_grad == nullptr, "Nonlinear operator gradient must be a HypreParMatrix");
  superlu_grad_mat_.emplace(*matr_grad);
  return *superlu_grad_mat_;
}

void EquationSolver::DefineInputFileSchema(axom::inlet::Table& table)
{
  auto& linear_table = table.addStruct("linear", "Linear Equation Solver Parameters")
                           .required()
                           .registerVerifier([](const axom::inlet::Table& table_to_verify) {
                             // Make sure that the provided options match the desired linear solver type
                             const bool is_iterative = (table_to_verify["type"].get<std::string>() == "iterative") &&
                                                       table_to_verify.contains("iterative_options");
                             const bool is_direct = (table_to_verify["type"].get<std::string>() == "direct") &&
                                                    table_to_verify.contains("direct_options");
                             return is_iterative || is_direct;
                           });

  // Enforce the solver type - must be iterative or direct
  linear_table.addString("type", "The type of solver parameters to use (iterative|direct)")
      .required()
      .validValues({"iterative", "direct"});

  auto& iterative_table = linear_table.addStruct("iterative_options", "Iterative solver parameters");
  iterative_table.addDouble("rel_tol", "Relative tolerance for the linear solve.").defaultValue(1.0e-6);
  iterative_table.addDouble("abs_tol", "Absolute tolerance for the linear solve.").defaultValue(1.0e-8);
  iterative_table.addInt("max_iter", "Maximum iterations for the linear solve.").defaultValue(5000);
  iterative_table.addInt("print_level", "Linear print level.").defaultValue(0);
  iterative_table.addString("solver_type", "Solver type (gmres|minres|cg).").defaultValue("gmres");
  iterative_table.addString("prec_type", "Preconditioner type (JacobiSmoother|L1JacobiSmoother|AMG|BlockILU).")
      .defaultValue("JacobiSmoother");

  auto& direct_table = linear_table.addStruct("direct_options", "Direct solver parameters");
  direct_table.addInt("print_level", "Linear print level.").defaultValue(0);

  // Only needed for nonlinear problems
  auto& nonlinear_table = table.addStruct("nonlinear", "Newton Equation Solver Parameters").required(false);
  nonlinear_table.addDouble("rel_tol", "Relative tolerance for the Newton solve.").defaultValue(1.0e-2);
  nonlinear_table.addDouble("abs_tol", "Absolute tolerance for the Newton solve.").defaultValue(1.0e-4);
  nonlinear_table.addInt("max_iter", "Maximum iterations for the Newton solve.").defaultValue(500);
  nonlinear_table.addInt("print_level", "Nonlinear print level.").defaultValue(0);
  nonlinear_table.addString("solver_type", "Solver type (MFEMNewton|KINFullStep|KINLineSearch)")
      .defaultValue("MFEMNewton");
}

}  // namespace serac::mfem_ext

using serac::LinearSolverOptions;
using serac::NonlinearSolverOptions;
using serac::mfem_ext::EquationSolver;

LinearSolverOptions FromInlet<LinearSolverOptions>::operator()(const axom::inlet::Table& base)
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
      std::string msg = fmt::format("Unknown Linear solver type given: {0}", solver_type);
      SLIC_ERROR(msg);
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
      std::string msg = fmt::format("Unknown preconditioner type given: {0}", prec_type);
      SLIC_ERROR(msg);
    }
    options = iter_options;
  } else if (type == "direct") {
    serac::DirectSolverOptions direct_options;
    direct_options.print_level = base["direct_options/print_level"];
    options                    = direct_options;
  }
  return options;
}

NonlinearSolverOptions FromInlet<NonlinearSolverOptions>::operator()(const axom::inlet::Table& base)
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
    SLIC_ERROR(fmt::format("Unknown nonlinear solver type given: {0}", solver_type));
  }
  return options;
}

EquationSolver FromInlet<EquationSolver>::operator()(const axom::inlet::Table& base)
{
  auto lin = base["linear"].get<LinearSolverOptions>();
  if (base.contains("nonlinear")) {
    auto nonlin = base["nonlinear"].get<NonlinearSolverOptions>();
    return EquationSolver(MPI_COMM_WORLD, lin, nonlin);
  }
  return EquationSolver(MPI_COMM_WORLD, lin);
}

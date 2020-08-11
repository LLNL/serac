// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef SERAC_TYPES
#define SERAC_TYPES

#include <memory>
#include <variant>

#include "mfem.hpp"

namespace serac {

// Option bundling enums

enum class OutputType
{
  GLVis,
  VisIt
};

enum class TimestepMethod
{
  BackwardEuler,
  SDIRK33,
  ForwardEuler,
  RK2,
  RK3SSP,
  RK4,
  GeneralizedAlpha,
  ImplicitMidpoint,
  SDIRK23,
  SDIRK34,
  QuasiStatic
};

enum class LinearSolver
{
  CG,
  GMRES,
  MINRES
};

enum class Preconditioner
{
  Jacobi,
  BoomerAMG
};

enum class CouplingScheme
{
  OperatorSplit,
  FixedPoint,
  FullyCoupled
};

// Parameter bundles

struct LinearSolverParameters {
  double         rel_tol;
  double         abs_tol;
  int            print_level;
  int            max_iter;
  LinearSolver   lin_solver;
  Preconditioner prec;
};

struct NonlinearSolverParameters {
  double rel_tol;
  double abs_tol;
  int    max_iter;
  int    print_level;
};

// Finite element information bundle
struct FiniteElementState {
  std::shared_ptr<mfem::ParFiniteElementSpace>   space;
  std::shared_ptr<mfem::FiniteElementCollection> coll;
  std::shared_ptr<mfem::ParGridFunction>         gf;
  std::shared_ptr<mfem::Vector>                  true_vec;
  std::shared_ptr<mfem::ParMesh>                 mesh;
  std::string                                    name = "";
};

// Boundary condition information
struct BoundaryCondition {
  using Coef = std::variant<std::shared_ptr<mfem::Coefficient>, std::shared_ptr<mfem::VectorCoefficient>>;
  mfem::Array<int>                      markers;
  mfem::Array<int>                      true_dofs;
  int                                   component;
  Coef                                  coef;
  std::unique_ptr<mfem::HypreParMatrix> eliminated_matrix_entries;
};

/**
 * Wraps a (currently iterative) system solver and handles the configuration of linear
 * or nonlinear solvers
 */
class SystemSolver : public mfem::Solver {
 public:
  // TODO: Eliminate this once a dependency injection approach is used for the solvers
  SystemSolver() = default;
  /**
   * Constructs a new solver wrapper
   * @param[in] comm The MPI communicator object
   * @param[in] lin_params The parameters for the linear solver
   * @param[in] nonlin_params The optional parameters for the optional nonlinear solver
   * @see serac::LinearSolverParameters
   * @see serac::NonlinearSolverParameters
   */
  SystemSolver(MPI_Comm comm, const LinearSolverParameters& lin_params,
               const std::optional<NonlinearSolverParameters>& nonlin_params = std::nullopt);

  /**
   * Sets a preconditioner for the underlying linear solver object
   * @param[in] prec The preconditioner, of which the object takes ownership
   * @note The preconditioner must be moved into the call
   * @code(.cpp)
   * solver.setPreconditioner(std::move(prec));
   * @endcode
   */
  void setPreconditioner(std::unique_ptr<mfem::Solver>&& prec)
  {
    prec_ = std::move(prec);
    iter_lin_solver_->SetPreconditioner(*prec_);
  }

  /** 
   * Updates the solver with the provided operator
   * @param[in] op The operator (system matrix) to use, "A" in Ax = b 
   * @note Implements mfem::Operator::SetOperator
   */
  void SetOperator(const mfem::Operator& op) override { solver().SetOperator(op); }

  /** 
   * Solves the system
   * @param[in] x The system's RHS vector, "b" in Ax = b
   * @param[out] y The system's solution vector, "x" in Ax = b
   * @note Implements mfem::Operator::Mult
   */
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override { solver().Mult(x, y); }

  /**
   * Returns the underlying solver object
   * @return The underlying nonlinear solver, if one was configured
   * when the object was constructed, otherwise, the underlying linear solver
   */
  mfem::IterativeSolver& solver() { return (nonlin_solver_) ? **nonlin_solver_ : *iter_lin_solver_; }
  const mfem::IterativeSolver& solver() const { return (nonlin_solver_) ? **nonlin_solver_ : *iter_lin_solver_; }

 private:
  std::unique_ptr<mfem::IterativeSolver>                iter_lin_solver_;
  std::optional<std::unique_ptr<mfem::IterativeSolver>> nonlin_solver_;
  std::unique_ptr<mfem::Solver>                         prec_;
};

}  // namespace serac

#endif

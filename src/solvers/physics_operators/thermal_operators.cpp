// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "thermal_operators.hpp"

#include "common/expr_template_ops.hpp"
#include "infrastructure/logger.hpp"

namespace serac {

DynamicConductionOperator::DynamicConductionOperator(mfem::ParFiniteElementSpace&           fe_space,
                                                     const serac::LinearSolverParameters&   params,
                                                     std::vector<serac::BoundaryCondition>& ess_bdr)
    : mfem::TimeDependentOperator(fe_space.GetTrueVSize(), 0.0),
      ess_bdr_(ess_bdr),
      z_(fe_space.GetTrueVSize()),
      y_(fe_space.GetTrueVSize()),
      x_(fe_space.GetTrueVSize()),
      old_dt_(-1.0)
{
  // Set the mass solver options (CG and Jacobi for now)
  M_inv_ = EquationSolver(fe_space.GetComm(), params);

  M_inv_.solver().iterative_mode = false;
  auto M_prec                    = std::make_unique<mfem::HypreSmoother>();
  M_prec->SetType(mfem::HypreSmoother::Jacobi);
  M_inv_.SetPreconditioner(std::move(M_prec));

  // Use the same options for the T (= M + dt K) solver
  T_inv_ = EquationSolver(fe_space.GetComm(), params);

  T_inv_.solver().iterative_mode = false;

  auto T_prec = std::make_unique<mfem::HypreSmoother>();
  T_inv_.SetPreconditioner(std::move(T_prec));

  state_gf_ = std::make_unique<mfem::ParGridFunction>(&fe_space);
  bc_rhs_   = std::make_unique<mfem::Vector>(fe_space.GetTrueVSize());
}

void DynamicConductionOperator::setMatrices(const mfem::HypreParMatrix* M_mat, mfem::HypreParMatrix* K_mat)
{
  M_ = M_mat;
  K_ = K_mat;
  M_inv_.SetOperator(*M_);
}

void DynamicConductionOperator::setLoadVector(const mfem::Vector* rhs) { rhs_ = rhs; }

// TODO: allow for changing thermal essential boundary conditions
void DynamicConductionOperator::Mult(const mfem::Vector& u, mfem::Vector& du_dt) const
{
  SLIC_ASSERT_MSG(M_ != nullptr, "Mass matrix not set in ConductionSolver::Mult!");
  SLIC_ASSERT_MSG(K_ != nullptr, "Stiffness matrix not set in ConductionSolver::Mult!");

  y_ = u;

  *bc_rhs_ = *rhs_;
  for (auto& bc : ess_bdr_) {
    bc.eliminateToRHS(*K_, u, *bc_rhs_);
  }

  // Compute:
  //    du_dt = M^{-1}*-K(u)
  // for du_dt
  du_dt = M_inv_ * (-(*K_ * u) + *bc_rhs_);
}

void DynamicConductionOperator::ImplicitSolve(const double dt, const mfem::Vector& u, mfem::Vector& du_dt)
{
  SLIC_ASSERT_MSG(M_ != nullptr, "Mass matrix not set in ConductionSolver::ImplicitSolve!");
  SLIC_ASSERT_MSG(K_ != nullptr, "Stiffness matrix not set in ConductionSolver::ImplicitSolve!");

  // Save a copy of the current state vector
  y_ = u;

  // Solve the equation:
  //    du_dt = T^{-1}*[-K(u + dt*du_dt)]
  // for du_dt
  if (dt != old_dt_) {
    // T = M + dt K
    T_.reset(mfem::Add(1.0, *M_, dt, *K_));

    // Eliminate the essential DOFs from the T matrix
    for (auto& bc : ess_bdr_) {
      T_e_mat_.reset(T_->EliminateRowsCols(bc.getTrueDofs()));
    }
    T_inv_.SetOperator(*T_);
  }

  // Apply the boundary conditions
  *bc_rhs_ = *rhs_;
  x_       = 0.0;

  for (auto& bc : ess_bdr_) {
    bc.projectBdr(*state_gf_, t);
    state_gf_->SetFromTrueDofs(y_);
    state_gf_->GetTrueDofs(y_);
    bc.eliminateToRHS(*K_, y_, *bc_rhs_);
  }

  du_dt = T_inv_ * (-(*K_ * u) + *bc_rhs_);

  // Save the dt used to compute the T matrix
  old_dt_ = dt;
}

DynamicConductionOperator::~DynamicConductionOperator() {}

}  // namespace serac

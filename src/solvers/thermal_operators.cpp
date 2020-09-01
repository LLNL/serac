// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "thermal_operators.hpp"

#include "common/logger.hpp"

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
  M_solver_ = EquationSolver(fe_space.GetComm(), params);

  M_solver_.linearSolver()->iterative_mode = false;
  auto M_prec                       = std::make_unique<mfem::HypreSmoother>();
  M_prec->SetType(mfem::HypreSmoother::Jacobi);
  M_solver_.SetPreconditioner(std::move(M_prec));

  // Use the same options for the T (= M + dt K) solver
  T_solver_ = EquationSolver(fe_space.GetComm(), params);

  T_solver_.linearSolver()->iterative_mode = false;

  auto T_prec = std::make_unique<mfem::HypreSmoother>();
  T_solver_.SetPreconditioner(std::move(T_prec));

  state_gf_ = std::make_unique<mfem::ParGridFunction>(&fe_space);
  bc_rhs_   = std::make_unique<mfem::Vector>(fe_space.GetTrueVSize());
}

void DynamicConductionOperator::setMatrices(const mfem::HypreParMatrix* M_mat, mfem::HypreParMatrix* K_mat)
{
  M_mat_ = M_mat;
  K_mat_ = K_mat;
  M_solver_.SetOperator(*M_mat_);
}

void DynamicConductionOperator::setLoadVector(const mfem::Vector* rhs) { rhs_ = rhs; }

// TODO: allow for changing thermal essential boundary conditions
void DynamicConductionOperator::Mult(const mfem::Vector& u, mfem::Vector& du_dt) const
{
  SLIC_ASSERT_MSG(M_mat_ != nullptr, "Mass matrix not set in ConductionSolver::Mult!");
  SLIC_ASSERT_MSG(K_mat_ != nullptr, "Stiffness matrix not set in ConductionSolver::Mult!");

  y_ = u;

  *bc_rhs_ = *rhs_;
  for (auto& bc : ess_bdr_) {
    bc.eliminateToRHS(*K_mat_, y_, *bc_rhs_);
  }

  // Compute:
  //    du_dt = M^{-1}*-K(u)
  // for du_dt
  K_mat_->Mult(y_, z_);
  z_.Neg();  // z = -zw z_.Add(1.0, *bc_rhs_);
  z_.Add(1.0, *bc_rhs_);
  M_solver_.Mult(z_, du_dt);
}

void DynamicConductionOperator::ImplicitSolve(const double dt, const mfem::Vector& u, mfem::Vector& du_dt)
{
  SLIC_ASSERT_MSG(M_mat_ != nullptr, "Mass matrix not set in ConductionSolver::ImplicitSolve!");
  SLIC_ASSERT_MSG(K_mat_ != nullptr, "Stiffness matrix not set in ConductionSolver::ImplicitSolve!");

  // Save a copy of the current state vector
  y_ = u;

  // Solve the equation:
  //    du_dt = M^{-1}*[-K(u + dt*du_dt)]
  // for du_dt
  if (dt != old_dt_) {
    T_mat_.reset(mfem::Add(1.0, *M_mat_, dt, *K_mat_));

    // Eliminate the essential DOFs from the T matrix
    for (auto& bc : ess_bdr_) {
      T_e_mat_.reset(T_mat_->EliminateRowsCols(bc.getTrueDofs()));
    }
    T_solver_.SetOperator(*T_mat_);
  }

  // Apply the boundary conditions
  *bc_rhs_ = *rhs_;
  x_       = 0.0;

  for (auto& bc : ess_bdr_) {
    bc.projectBdr(*state_gf_, t);
    state_gf_->SetFromTrueDofs(y_);
    state_gf_->GetTrueDofs(y_);
    bc.eliminateToRHS(*K_mat_, y_, *bc_rhs_);
  }
  K_mat_->Mult(y_, z_);
  z_.Neg();
  z_.Add(1.0, *bc_rhs_);
  T_solver_.Mult(z_, du_dt);

  // Save the dt used to compute the T matrix
  old_dt_ = dt;
}

DynamicConductionOperator::~DynamicConductionOperator() {}

}  // namespace serac

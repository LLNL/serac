// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "thermal_operators.hpp"

#include "common/logger.hpp"

DynamicConductionOperator::DynamicConductionOperator(
    std::shared_ptr<mfem::ParFiniteElementSpace> fespace, const serac::LinearSolverParameters &params,
    const std::vector<std::shared_ptr<serac::BoundaryCondition> > &ess_bdr)
    : mfem::TimeDependentOperator(fespace->GetTrueVSize(), 0.0),
      fespace_(fespace),
      ess_bdr_(ess_bdr),
      z_(fespace->GetTrueVSize()),
      y_(fespace->GetTrueVSize()),
      x_(fespace->GetTrueVSize()),
      old_dt_(-1.0)
{
  // Set the mass solver options (CG and Jacobi for now)
  M_solver_ = std::make_unique<mfem::CGSolver>(fespace_->GetComm());
  M_prec_   = std::make_unique<mfem::HypreSmoother>();

  M_solver_->iterative_mode = false;
  M_solver_->SetRelTol(params.rel_tol);
  M_solver_->SetAbsTol(params.abs_tol);
  M_solver_->SetMaxIter(params.max_iter);
  M_solver_->SetPrintLevel(params.print_level);
  M_prec_->SetType(mfem::HypreSmoother::Jacobi);
  M_solver_->SetPreconditioner(*M_prec_);

  // Use the same options for the T (= M + dt K) solver
  T_solver_ = std::make_unique<mfem::CGSolver>(fespace_->GetComm());
  T_prec_   = std::make_unique<mfem::HypreSmoother>();

  T_solver_->iterative_mode = false;
  T_solver_->SetRelTol(params.rel_tol);
  T_solver_->SetAbsTol(params.abs_tol);
  T_solver_->SetMaxIter(params.max_iter);
  T_solver_->SetPrintLevel(params.print_level);
  T_solver_->SetPreconditioner(*T_prec_);

  state_gf_ = std::make_shared<mfem::ParGridFunction>(fespace_.get());
  bc_rhs_   = std::make_shared<mfem::Vector>(fespace->GetTrueVSize());
}

void DynamicConductionOperator::SetMatrices(std::shared_ptr<mfem::HypreParMatrix> M_mat,
                                            std::shared_ptr<mfem::HypreParMatrix> K_mat)
{
  M_mat_ = M_mat;
  m_K_mat = K_mat;
}

void DynamicConductionOperator::SetLoadVector(std::shared_ptr<mfem::Vector> rhs) { rhs_ = rhs; }

// TODO: allow for changing thermal essential boundary conditions
void DynamicConductionOperator::Mult(const mfem::Vector &u, mfem::Vector &du_dt) const
{
  SLIC_ASSERT_MSG(M_mat_ != nullptr, "Mass matrix not set in ConductionSolver::Mult!");
  SLIC_ASSERT_MSG(m_K_mat != nullptr, "Stiffness matrix not set in ConductionSolver::Mult!");

  y_ = u;
  M_solver_->SetOperator(*M_mat_);

  *bc_rhs_ = *rhs_;
  for (auto &bc : ess_bdr_) {
    mfem::EliminateBC(*m_K_mat, *bc->eliminated_matrix_entries, bc->true_dofs, y_, *bc_rhs_);
  }

  // Compute:
  //    du_dt = M^{-1}*-K(u)
  // for du_dt
  m_K_mat->Mult(y_, z_);
  z_.Neg();  // z = -zw  m_z.Add(1.0, *m_bc_rhs);
  z_.Add(1.0, *bc_rhs_);
  M_solver_->Mult(z_, du_dt);
}

void DynamicConductionOperator::ImplicitSolve(const double dt, const mfem::Vector &u, mfem::Vector &du_dt)
{
  SLIC_ASSERT_MSG(M_mat_ != nullptr, "Mass matrix not set in ConductionSolver::ImplicitSolve!");
  SLIC_ASSERT_MSG(m_K_mat != nullptr, "Stiffness matrix not set in ConductionSolver::ImplicitSolve!");

  // Save a copy of the current state vector
  y_ = u;

  // Solve the equation:
  //    du_dt = M^{-1}*[-K(u + dt*du_dt)]
  // for du_dt
  if (dt != old_dt_) {
    T_mat_.reset(mfem::Add(1.0, *M_mat_, dt, *m_K_mat));

    // Eliminate the essential DOFs from the T matrix
    for (auto &bc : ess_bdr_) {
      T_e_mat_.reset(T_mat_->EliminateRowsCols(bc->true_dofs));
    }
    T_solver_->SetOperator(*T_mat_);
  }

  // Apply the boundary conditions
  *bc_rhs_ = *rhs_;
  x_       = 0.0;

  for (auto &bc : ess_bdr_) {
    if (bc->scalar_coef != nullptr) {
      bc->scalar_coef->SetTime(t);
      state_gf_->SetFromTrueDofs(y_);
      state_gf_->ProjectBdrCoefficient(*bc->scalar_coef, bc->markers);
      state_gf_->GetTrueDofs(y_);

      mfem::EliminateBC(*m_K_mat, *bc->eliminated_matrix_entries, bc->true_dofs, y_, *bc_rhs_);
    }
  }

  m_K_mat->Mult(y_, z_);
  z_.Neg();
  z_.Add(1.0, *bc_rhs_);
  T_solver_->Mult(z_, du_dt);

  // Save the dt used to compute the T matrix
  old_dt_ = dt;
}

DynamicConductionOperator::~DynamicConductionOperator() {}

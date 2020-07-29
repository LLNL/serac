// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "nonlinear_solid_operators.hpp"

#include "common/logger.hpp"

NonlinearSolidQuasiStaticOperator::NonlinearSolidQuasiStaticOperator(std::shared_ptr<mfem::ParNonlinearForm> H_form)
    : mfem::Operator(H_form->FESpace()->GetTrueVSize()), m_H_form(H_form) {}

// compute: y = H(x,p)
void NonlinearSolidQuasiStaticOperator::Mult(const mfem::Vector &k, mfem::Vector &y) const
{
  // Apply the nonlinear form
  m_H_form->Mult(k, y);
}

// Compute the Jacobian from the nonlinear form
mfem::Operator &NonlinearSolidQuasiStaticOperator::GetGradient(const mfem::Vector &x) const
{
  return m_H_form->GetGradient(x);
}

// destructor
NonlinearSolidQuasiStaticOperator::~NonlinearSolidQuasiStaticOperator() {}

NonlinearSolidDynamicOperator::NonlinearSolidDynamicOperator(
    std::shared_ptr<mfem::ParNonlinearForm> H_form, std::shared_ptr<mfem::ParBilinearForm> S_form,
    std::shared_ptr<mfem::ParBilinearForm> M_form, const std::vector<std::shared_ptr<serac::BoundaryCondition> > &ess_bdr,
    mfem::NewtonSolver &newton_solver, const serac::LinearSolverParameters &lin_params)
    : mfem::TimeDependentOperator(M_form->ParFESpace()->TrueVSize() * 2),
      m_M_form(M_form),
      m_S_form(S_form),
      m_H_form(H_form),
      m_newton_solver(newton_solver),
      m_ess_bdr(ess_bdr),
      m_lin_params(lin_params),
      m_z(height / 2)
{
  // Assemble the mass matrix and eliminate the fixed DOFs
  m_M_mat.reset(m_M_form->ParallelAssemble());
  for (auto &bc : m_ess_bdr) {
    auto Me = std::unique_ptr<mfem::HypreParMatrix>(m_M_mat->EliminateRowsCols(bc->true_dofs));
  }

  // Set the mass matrix solver options
  m_M_solver.iterative_mode = false;
  m_M_solver.SetRelTol(m_lin_params.rel_tol);
  m_M_solver.SetAbsTol(m_lin_params.abs_tol);
  m_M_solver.SetMaxIter(m_lin_params.max_iter);
  m_M_solver.SetPrintLevel(m_lin_params.print_level);
  m_M_prec.SetType(mfem::HypreSmoother::Jacobi);
  m_M_solver.SetPreconditioner(m_M_prec);
  m_M_solver.SetOperator(*m_M_mat);

  // Construct the reduced system operator and initialize the newton solver with
  // it
  m_reduced_oper = std::make_unique<NonlinearSolidReducedSystemOperator>(H_form, S_form, M_form, m_ess_bdr);
  m_newton_solver.SetOperator(*m_reduced_oper);
}

void NonlinearSolidDynamicOperator::Mult(const mfem::Vector &vx, mfem::Vector &dvx_dt) const
{
  // Create views to the sub-vectors v, x of vx, and dv_dt, dx_dt of dvx_dt
  int          sc = height / 2;
  mfem::Vector v(vx.GetData() + 0, sc);
  mfem::Vector x(vx.GetData() + sc, sc);
  mfem::Vector dv_dt(dvx_dt.GetData() + 0, sc);
  mfem::Vector dx_dt(dvx_dt.GetData() + sc, sc);

  m_H_form->Mult(x, m_z);
  m_S_form->TrueAddMult(v, m_z);
  for (auto &bc : m_ess_bdr) {
    m_z.SetSubVector(bc->true_dofs, 0.0);
  }
  m_z.Neg();  // z = -z
  m_M_solver.Mult(m_z, dv_dt);

  dx_dt = v;
}

void NonlinearSolidDynamicOperator::ImplicitSolve(const double dt, const mfem::Vector &vx, mfem::Vector &dvx_dt)
{
  int          sc = height / 2;
  mfem::Vector v(vx.GetData() + 0, sc);
  mfem::Vector x(vx.GetData() + sc, sc);
  mfem::Vector dv_dt(dvx_dt.GetData() + 0, sc);
  mfem::Vector dx_dt(dvx_dt.GetData() + sc, sc);

  // By eliminating kx from the coupled system:
  //    kv = -M^{-1}*[H(x + dt*kx) + S*(v + dt*kv)]
  //    kx = v + dt*kv
  // we reduce it to a nonlinear equation for kv, represented by the
  // m_reduced_oper. This equation is solved with the m_newton_solver
  // object (using m_J_solver and m_J_prec internally).
  m_reduced_oper->SetParameters(dt, &v, &x);
  mfem::Vector zero;  // empty vector is interpreted as zero r.h.s. by NewtonSolver
  m_newton_solver.Mult(zero, dv_dt);
  SLIC_WARNING_IF(m_newton_solver.GetConverged(), "Newton solver did not converge.");
  add(v, dt, dv_dt, dx_dt);
}

// destructor
NonlinearSolidDynamicOperator::~NonlinearSolidDynamicOperator() {}

NonlinearSolidReducedSystemOperator::NonlinearSolidReducedSystemOperator(
    std::shared_ptr<mfem::ParNonlinearForm> H_form, std::shared_ptr<mfem::ParBilinearForm> S_form,
    std::shared_ptr<mfem::ParBilinearForm> M_form, const std::vector<std::shared_ptr<serac::BoundaryCondition> > &ess_bdr)
    : mfem::Operator(M_form->ParFESpace()->TrueVSize()),
      m_M_form(M_form),
      m_S_form(S_form),
      m_H_form(H_form),
      m_dt(0.0),
      m_v(nullptr),
      m_x(nullptr),
      m_w(height),
      m_z(height),
      m_ess_bdr(ess_bdr)
{
}

void NonlinearSolidReducedSystemOperator::SetParameters(double dt, const mfem::Vector *v, const mfem::Vector *x)
{
  m_dt = dt;
  m_v  = v;
  m_x  = x;
}

void NonlinearSolidReducedSystemOperator::Mult(const mfem::Vector &k, mfem::Vector &y) const
{
  // compute: y = H(x + dt*(v + dt*k)) + M*k + S*(v + dt*k)
  add(*m_v, m_dt, k, m_w);
  add(*m_x, m_dt, m_w, m_z);
  m_H_form->Mult(m_z, y);
  m_M_form->TrueAddMult(k, y);
  m_S_form->TrueAddMult(m_w, y);
  for (auto &bc : m_ess_bdr) {
    y.SetSubVector(bc->true_dofs, 0.0);
  }
}

mfem::Operator &NonlinearSolidReducedSystemOperator::GetGradient(const mfem::Vector &k) const
{
  // Form the gradient of the complete nonlinear operator
  auto localJ = std::unique_ptr<mfem::SparseMatrix>(Add(1.0, m_M_form->SpMat(), m_dt, m_S_form->SpMat()));
  add(*m_v, m_dt, k, m_w);
  add(*m_x, m_dt, m_w, m_z);
  localJ->Add(m_dt * m_dt, m_H_form->GetLocalGradient(m_z));
  m_jacobian.reset(m_M_form->ParallelAssemble(localJ.get()));

  // Eliminate the fixed boundary DOFs
  //
  // This call eliminates the appropriate DOFs in m_jacobian and returns the
  // eliminated DOFs in Je. We don't need this so it gets deleted.
  for (auto &bc : m_ess_bdr) {
    auto Je = std::unique_ptr<mfem::HypreParMatrix>(m_jacobian->EliminateRowsCols(bc->true_dofs));
  }
  return *m_jacobian;
}

NonlinearSolidReducedSystemOperator::~NonlinearSolidReducedSystemOperator() {}

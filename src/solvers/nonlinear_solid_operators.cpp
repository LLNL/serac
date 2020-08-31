// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "nonlinear_solid_operators.hpp"

#include "common/expr_template_ops.hpp"
#include "common/logger.hpp"

namespace serac {

NonlinearSolidQuasiStaticOperator::NonlinearSolidQuasiStaticOperator(std::unique_ptr<mfem::ParNonlinearForm> H_form)
    : mfem::Operator(H_form->FESpace()->GetTrueVSize()), H_form_(std::move(H_form))
{
}

// compute: y = H(x,p)
void NonlinearSolidQuasiStaticOperator::Mult(const mfem::Vector& k, mfem::Vector& y) const
{
  // Apply the nonlinear form H_form_->Mult(k, y);
  H_form_->Mult(k, y);
}

// Compute the Jacobian from the nonlinear form
mfem::Operator& NonlinearSolidQuasiStaticOperator::GetGradient(const mfem::Vector& x) const
{
  return H_form_->GetGradient(x);
}

// destructor
NonlinearSolidQuasiStaticOperator::~NonlinearSolidQuasiStaticOperator() {}

NonlinearSolidDynamicOperator::NonlinearSolidDynamicOperator(std::unique_ptr<mfem::ParNonlinearForm>      H_form,
                                                             std::unique_ptr<mfem::ParBilinearForm>       S_form,
                                                             std::unique_ptr<mfem::ParBilinearForm>       M_form,
                                                             const std::vector<serac::BoundaryCondition>& ess_bdr,
                                                             mfem::IterativeSolver&                       newton_solver,
                                                             const serac::LinearSolverParameters&         lin_params)
    : mfem::TimeDependentOperator(M_form->ParFESpace()->TrueVSize() * 2),
      M_form_(std::move(M_form)),
      S_form_(std::move(S_form)),
      H_form_(std::move(H_form)),
      newton_solver_(newton_solver),
      ess_bdr_(ess_bdr),
      lin_params_(lin_params),
      z_(height / 2)
{
  // Assemble the mass matrix and eliminate the fixed DOFs
  M_mat_.reset(M_form_->ParallelAssemble());
  for (auto& bc : ess_bdr_) {
    auto Me = std::unique_ptr<mfem::HypreParMatrix>(M_mat_->EliminateRowsCols(bc.getTrueDofs()));
  }

  M_solver_ = EquationSolver(H_form_->ParFESpace()->GetComm(), lin_params);

  auto M_prec                       = std::make_unique<mfem::HypreSmoother>();
  M_solver_.solver().iterative_mode = false;

  M_prec->SetType(mfem::HypreSmoother::Jacobi);
  M_solver_.SetPreconditioner(std::move(M_prec));

  M_solver_.SetOperator(*M_mat_);

  // Construct the reduced system operator and initialize the newton solver with
  // it
  reduced_oper_ = std::make_unique<NonlinearSolidReducedSystemOperator>(*H_form_, *S_form_, *M_form_, ess_bdr_);
  newton_solver_.SetOperator(*reduced_oper_);
}

void NonlinearSolidDynamicOperator::Mult(const mfem::Vector& vx, mfem::Vector& dvx_dt) const
{
  // Create views to the sub-vectors v, x of vx, and dv_dt, dx_dt of dvx_dt
  int          sc = height / 2;
  mfem::Vector v(vx.GetData() + 0, sc);
  mfem::Vector x(vx.GetData() + sc, sc);
  mfem::Vector dv_dt(dvx_dt.GetData() + 0, sc);
  mfem::Vector dx_dt(dvx_dt.GetData() + sc, sc);

  z_ = *H_form_ * x;
  S_form_->TrueAddMult(v, z_);
  for (auto& bc : ess_bdr_) {
    z_.SetSubVector(bc.getTrueDofs(), 0.0);
  }

  dv_dt = M_solver_ * -z_;
  dx_dt = v;
}

void NonlinearSolidDynamicOperator::ImplicitSolve(const double dt, const mfem::Vector& vx, mfem::Vector& dvx_dt)
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
  reduced_oper_->SetParameters(dt, &v, &x);
  mfem::Vector zero;  // empty vector is interpreted as zero r.h.s. by NewtonSolver
  dv_dt = newton_solver_ * zero;
  SLIC_WARNING_IF(newton_solver_.GetConverged(), "Newton solver did not converge.");
  dx_dt = v + (dt * dv_dt);
}

// destructor
NonlinearSolidDynamicOperator::~NonlinearSolidDynamicOperator() {}

NonlinearSolidReducedSystemOperator::NonlinearSolidReducedSystemOperator(
    const mfem::ParNonlinearForm& H_form, const mfem::ParBilinearForm& S_form, mfem::ParBilinearForm& M_form,
    const std::vector<serac::BoundaryCondition>& ess_bdr)
    : mfem::Operator(M_form.ParFESpace()->TrueVSize()),
      M_form_(M_form),
      S_form_(S_form),
      H_form_(H_form),
      dt_(0.0),
      v_(nullptr),
      x_(nullptr),
      w_(height),
      z_(height),
      ess_bdr_(ess_bdr)
{
}

void NonlinearSolidReducedSystemOperator::SetParameters(double dt, const mfem::Vector* v, const mfem::Vector* x)
{
  dt_ = dt;
  v_  = v;
  x_  = x;
}

void NonlinearSolidReducedSystemOperator::Mult(const mfem::Vector& k, mfem::Vector& y) const
{
  // compute: y = H(x + dt*(v + dt*k)) + M*k + S*(v + dt*k)
  w_ = *v_ + (dt_ * k);
  z_ = *x_ + (dt_ * w_);
  y  = H_form_ * z_;
  M_form_.TrueAddMult(k, y);
  S_form_.TrueAddMult(w_, y);
  for (const auto& bc : ess_bdr_) {
    y.SetSubVector(bc.getTrueDofs(), 0.0);
  }
}

mfem::Operator& NonlinearSolidReducedSystemOperator::GetGradient(const mfem::Vector& k) const
{
  // Form the gradient of the complete nonlinear operator
  auto localJ = std::unique_ptr<mfem::SparseMatrix>(Add(1.0, M_form_.SpMat(), dt_, S_form_.SpMat()));
  w_          = *v_ + (dt_ * k);
  z_          = *x_ + (dt_ * w_);
  localJ->Add(dt_ * dt_, H_form_.GetLocalGradient(z_));
  jacobian_.reset(M_form_.ParallelAssemble(localJ.get()));

  // Eliminate the fixed boundary DOFs
  //
  // This call eliminates the appropriate DOFs in jacobian_ and returns the
  // eliminated DOFs in Je. We don't need this so it gets deleted.
  for (auto& bc : ess_bdr_) {
    auto Je = std::unique_ptr<mfem::HypreParMatrix>(jacobian_->EliminateRowsCols(bc.getTrueDofs()));
  }
  return *jacobian_;
}

NonlinearSolidReducedSystemOperator::~NonlinearSolidReducedSystemOperator() {}

}  // namespace serac

// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "elasticity_solver.hpp"

#include "common/logger.hpp"

const int num_fields = 1;

ElasticitySolver::ElasticitySolver(int order, std::shared_ptr<mfem::ParMesh> pmesh)
    : BaseSolver(pmesh->GetComm(), num_fields, order),
      displacement_(state_[0]),
      K_form_(nullptr),
      l_form_(nullptr),
      K_mat_(nullptr),
      K_e_mat_(nullptr),
      rhs_(nullptr),
      bc_rhs_(nullptr),
      K_solver_(nullptr),
      K_prec_(nullptr),
      mu_(nullptr),
      lambda_(nullptr),
      body_force_(nullptr)
{
  pmesh->EnsureNodes();
  displacement_->mesh = pmesh;
  displacement_->coll = std::make_shared<mfem::H1_FECollection>(order, pmesh->Dimension(), mfem::Ordering::byVDIM);
  displacement_->space =
      std::make_shared<mfem::ParFiniteElementSpace>(pmesh.get(), displacement_->coll.get(), pmesh->Dimension());
  displacement_->gf       = std::make_shared<mfem::ParGridFunction>(displacement_->space.get());
  displacement_->true_vec = std::make_shared<mfem::HypreParVector>(displacement_->space.get());

  // and initial conditions
  *displacement_->gf       = 0.0;
  *displacement_->true_vec = 0.0;

  displacement_->name = "displacement";
}

void ElasticitySolver::SetDisplacementBCs(std::set<int> &                          disp_bdr,
                                          std::shared_ptr<mfem::VectorCoefficient> disp_bdr_coef, int component)
{
  SetEssentialBCs(disp_bdr, disp_bdr_coef, *displacement_->space, component);
}

void ElasticitySolver::SetTractionBCs(std::set<int> &trac_bdr, std::shared_ptr<mfem::VectorCoefficient> trac_bdr_coef,
                                      int component)
{
  SetNaturalBCs(trac_bdr, trac_bdr_coef, component);
}

void ElasticitySolver::SetLameParameters(mfem::Coefficient &lambda, mfem::Coefficient &mu)
{
  lambda_ = &lambda;
  mu_     = &mu;
}

void ElasticitySolver::SetBodyForce(mfem::VectorCoefficient &force) { body_force_ = &force; }

void ElasticitySolver::SetLinearSolverParameters(const serac::LinearSolverParameters &params) { lin_params_ = params; }

void ElasticitySolver::CompleteSetup()
{
  SLIC_ASSERT_MSG(mu_ != nullptr, "Lame mu not set in ElasticitySolver!");
  SLIC_ASSERT_MSG(lambda_ != nullptr, "Lame lambda not set in ElasticitySolver!");

  // Define the parallel bilinear form
  K_form_ = new mfem::ParBilinearForm(displacement_->space.get());

  // Add the elastic integrator
  K_form_->AddDomainIntegrator(new mfem::ElasticityIntegrator(*lambda_, *mu_));
  K_form_->Assemble();
  K_form_->Finalize();

  // Define the parallel linear form

  l_form_ = new mfem::ParLinearForm(displacement_->space.get());

  // Add the traction integrator
  if (nat_bdr_.size() > 0) {
    for (auto &nat_bc : nat_bdr_) {
      l_form_->AddBoundaryIntegrator(new mfem::VectorBoundaryLFIntegrator(*nat_bc->vec_coef), nat_bc->markers);
    }
    l_form_->Assemble();
    rhs_ = l_form_->ParallelAssemble();
  } else {
    rhs_  = new mfem::HypreParVector(displacement_->space.get());
    *rhs_ = 0.0;
  }

  // Assemble the stiffness matrix
  K_mat_ = K_form_->ParallelAssemble();

  // Eliminate the essential DOFs
  for (auto &bc : ess_bdr_) {
    K_e_mat_ = K_mat_->EliminateRowsCols(bc->true_dofs);
  }

  // Initialize the eliminate BC RHS vector
  bc_rhs_  = new mfem::HypreParVector(displacement_->space.get());
  *bc_rhs_ = 0.0;

  // Initialize the true vector
  displacement_->gf->GetTrueDofs(*displacement_->true_vec);

  if (lin_params_.prec == serac::Preconditioner::BoomerAMG) {
    SLIC_WARNING_IF(displacement_->space->GetOrdering() == mfem::Ordering::byVDIM,
                    "Attempting to use BoomerAMG with nodal ordering.");

    mfem::HypreBoomerAMG *prec_amg = new mfem::HypreBoomerAMG();
    prec_amg->SetPrintLevel(lin_params_.print_level);
    prec_amg->SetElasticityOptions(displacement_->space.get());
    K_prec_ = prec_amg;

    mfem::GMRESSolver *K_gmres = new mfem::GMRESSolver(displacement_->space->GetComm());
    K_gmres->SetRelTol(lin_params_.rel_tol);
    K_gmres->SetAbsTol(lin_params_.abs_tol);
    K_gmres->SetMaxIter(lin_params_.max_iter);
    K_gmres->SetPrintLevel(lin_params_.print_level);
    K_gmres->SetPreconditioner(*K_prec_);
    K_solver_ = K_gmres;

  }
  // If not AMG, just MINRES with Jacobi smoothing
  else {
    mfem::HypreSmoother *K_hypreSmoother = new mfem::HypreSmoother;
    K_hypreSmoother->SetType(mfem::HypreSmoother::l1Jacobi);
    K_hypreSmoother->SetPositiveDiagonal(true);
    K_prec_ = K_hypreSmoother;

    mfem::MINRESSolver *K_minres = new mfem::MINRESSolver(displacement_->space->GetComm());
    K_minres->SetRelTol(lin_params_.rel_tol);
    K_minres->SetAbsTol(lin_params_.abs_tol);
    K_minres->SetMaxIter(lin_params_.max_iter);
    K_minres->SetPrintLevel(lin_params_.print_level);
    K_minres->SetPreconditioner(*K_prec_);
    K_solver_ = K_minres;
  }
}

void ElasticitySolver::AdvanceTimestep(double &)
{
  // Initialize the true vector
  displacement_->gf->GetTrueDofs(*displacement_->true_vec);

  if (timestepper_ == serac::TimestepMethod::QuasiStatic) {
    QuasiStaticSolve();
  } else {
    SLIC_ERROR_ROOT(mpi_rank_, "Only quasistatics implemented for linear elasticity!");
    serac::ExitGracefully(true);
  }

  // Distribute the shared DOFs
  displacement_->gf->SetFromTrueDofs(*displacement_->true_vec);
  cycle_ += 1;
}

// Solve the Quasi-static system
void ElasticitySolver::QuasiStaticSolve()
{
  // Apply the boundary conditions
  *bc_rhs_ = *rhs_;
  for (auto &bc : ess_bdr_) {
    bc->vec_coef->SetTime(time_);
    displacement_->gf->ProjectBdrCoefficient(*bc->vec_coef, bc->markers);
    displacement_->gf->GetTrueDofs(*displacement_->true_vec);
    mfem::EliminateBC(*K_mat_, *K_e_mat_, bc->true_dofs, *displacement_->true_vec, *bc_rhs_);
  }

  K_solver_->SetOperator(*K_mat_);

  K_solver_->Mult(*bc_rhs_, *displacement_->true_vec);
}

ElasticitySolver::~ElasticitySolver()
{
  delete K_form_;
  delete l_form_;
  delete K_mat_;
  delete K_e_mat_;
  delete rhs_;
  delete bc_rhs_;
  delete K_solver_;
  delete K_prec_;
}

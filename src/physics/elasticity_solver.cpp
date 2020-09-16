// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "elasticity_solver.hpp"

#include "infrastructure/logger.hpp"
#include "infrastructure/terminator.hpp"

namespace serac {

constexpr int NUM_FIELDS = 1;

ElasticitySolver::ElasticitySolver(int order, std::shared_ptr<mfem::ParMesh> mesh)
    : BaseSolver(mesh, NUM_FIELDS, order),
      displacement_(std::make_shared<FiniteElementState>(*mesh, FEStateOptions{.order = order, .name = "displacement"}))
{
  mesh->EnsureNodes();
  state_[0] = displacement_;
}

void ElasticitySolver::setDisplacementBCs(const std::set<int>&                     disp_bdr,
                                          std::shared_ptr<mfem::VectorCoefficient> disp_bdr_coef, const int component)
{
  bcs_.addEssential(disp_bdr, disp_bdr_coef, *displacement_, component);
}

void ElasticitySolver::setTractionBCs(const std::set<int>&                     trac_bdr,
                                      std::shared_ptr<mfem::VectorCoefficient> trac_bdr_coef, const int component)
{
  bcs_.addNatural(trac_bdr, trac_bdr_coef, component);
}

void ElasticitySolver::setLameParameters(mfem::Coefficient& lambda, mfem::Coefficient& mu)
{
  lambda_ = &lambda;
  mu_     = &mu;
}

void ElasticitySolver::setBodyForce(mfem::VectorCoefficient& force) { body_force_ = &force; }

void ElasticitySolver::setLinearSolverParameters(const serac::LinearSolverParameters& params) { lin_params_ = params; }

void ElasticitySolver::completeSetup()
{
  SLIC_ASSERT_MSG(mu_ != nullptr, "Lame mu not set in ElasticitySolver!");
  SLIC_ASSERT_MSG(lambda_ != nullptr, "Lame lambda not set in ElasticitySolver!");

  // Define the parallel bilinear form
  K_form_ = displacement_->createOnSpace<mfem::ParBilinearForm>();

  // Add the elastic integrator
  K_form_->AddDomainIntegrator(new mfem::ElasticityIntegrator(*lambda_, *mu_));
  K_form_->Assemble();
  K_form_->Finalize();

  // Define the parallel linear form

  l_form_ = displacement_->createOnSpace<mfem::ParLinearForm>();

  // Add the traction integrator
  if (bcs_.naturals().size() > 0) {
    for (auto& nat_bc : bcs_.naturals()) {
      l_form_->AddBoundaryIntegrator(new mfem::VectorBoundaryLFIntegrator(nat_bc.vectorCoefficient()),
                                     nat_bc.markers());
    }
    l_form_->Assemble();
    rhs_.reset(l_form_->ParallelAssemble());
  } else {
    rhs_  = displacement_->createOnSpace<mfem::HypreParVector>();
    *rhs_ = 0.0;
  }

  // Assemble the stiffness matrix
  K_mat_ = std::unique_ptr<mfem::HypreParMatrix>(K_form_->ParallelAssemble());

  // Eliminate the essential DOFs
  for (auto& bc : bcs_.essentials()) {
    bc.eliminateFromMatrix(*K_mat_);
  }

  // Initialize the eliminate BC RHS vector
  bc_rhs_  = displacement_->createOnSpace<mfem::HypreParVector>();
  *bc_rhs_ = 0.0;

  // Initialize the true vector
  displacement_->initializeTrueVec();

  solver_ = EquationSolver(displacement_->comm(), lin_params_);
  if (lin_params_.prec == serac::Preconditioner::BoomerAMG) {
    SLIC_WARNING_IF(displacement_->space().GetOrdering() == mfem::Ordering::byVDIM,
                    "Attempting to use BoomerAMG with nodal ordering.");

    auto prec_amg = std::make_unique<mfem::HypreBoomerAMG>();
    prec_amg->SetPrintLevel(lin_params_.print_level);
    prec_amg->SetElasticityOptions(&displacement_->space());
    solver_.SetPreconditioner(std::move(prec_amg));
  }
  // If not AMG, just MINRES with Jacobi smoothing
  else {
    auto K_hypreSmoother = std::make_unique<mfem::HypreSmoother>();
    K_hypreSmoother->SetType(mfem::HypreSmoother::l1Jacobi);
    K_hypreSmoother->SetPositiveDiagonal(true);
    solver_.SetPreconditioner(std::move(K_hypreSmoother));
  }
}

void ElasticitySolver::advanceTimestep(double&)
{
  // Initialize the true vector
  displacement_->initializeTrueVec();

  if (timestepper_ == serac::TimestepMethod::QuasiStatic) {
    QuasiStaticSolve();
  } else {
    SLIC_ERROR_ROOT(mpi_rank_, "Only quasistatics implemented for linear elasticity!");
    serac::exitGracefully(true);
  }

  // Distribute the shared DOFs
  displacement_->distributeSharedDofs();
  cycle_ += 1;
}

// Solve the Quasi-static system
void ElasticitySolver::QuasiStaticSolve()
{
  // Apply the boundary conditions
  *bc_rhs_ = *rhs_;
  for (auto& bc : bcs_.essentials()) {
    bool should_be_scalar = false;
    bc.apply(*K_mat_, *bc_rhs_, *displacement_, time_, should_be_scalar);
  }

  solver_.SetOperator(*K_mat_);

  solver_.Mult(*bc_rhs_, displacement_->trueVec());
}

ElasticitySolver::~ElasticitySolver() {}

}  // namespace serac

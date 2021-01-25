// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/elasticity.hpp"

#include "serac/infrastructure/logger.hpp"
#include "serac/infrastructure/terminator.hpp"

namespace serac {

constexpr int NUM_FIELDS = 1;

Elasticity::Elasticity(int order, std::shared_ptr<mfem::ParMesh> mesh, const LinearSolverOptions& options)
    : BasePhysics(mesh, NUM_FIELDS, order),
      displacement_(*mesh, FiniteElementState::Options{.order = order, .name = "displacement"})
{
  mesh->EnsureNodes();
  state_.push_back(displacement_);

  // If the user wants the AMG preconditioner with a linear solver, set the pfes to be the displacement
  const auto& augmented_options = mfem_ext::AugmentAMGForElasticity(options, displacement_.space());

  K_inv_          = mfem_ext::EquationSolver(mesh->GetComm(), augmented_options);
  is_quasistatic_ = true;
}

void Elasticity::setDisplacementBCs(const std::set<int>&                     disp_bdr,
                                    std::shared_ptr<mfem::VectorCoefficient> disp_bdr_coef, const int component)
{
  bcs_.addEssential(disp_bdr, disp_bdr_coef, displacement_, component);
}

void Elasticity::setTractionBCs(const std::set<int>& trac_bdr, std::shared_ptr<mfem::VectorCoefficient> trac_bdr_coef,
                                const int component)
{
  bcs_.addNatural(trac_bdr, trac_bdr_coef, component);
}

void Elasticity::setLameParameters(mfem::Coefficient& lambda, mfem::Coefficient& mu)
{
  lambda_ = &lambda;
  mu_     = &mu;
}

void Elasticity::setBodyForce(mfem::VectorCoefficient& force) { body_force_ = &force; }

void Elasticity::completeSetup()
{
  SLIC_ASSERT_MSG(mu_ != nullptr, "Lame mu not set in ElasticitySolver!");
  SLIC_ASSERT_MSG(lambda_ != nullptr, "Lame lambda not set in ElasticitySolver!");

  // Define the parallel bilinear form
  K_form_ = displacement_.createOnSpace<mfem::ParBilinearForm>();

  // Add the elastic integrator
  K_form_->AddDomainIntegrator(new mfem::ElasticityIntegrator(*lambda_, *mu_));
  K_form_->Assemble();
  K_form_->Finalize();

  // Define the parallel linear form

  l_form_ = displacement_.createOnSpace<mfem::ParLinearForm>();

  // Add the traction integrator
  if (bcs_.naturals().size() > 0) {
    for (auto& nat_bc : bcs_.naturals()) {
      l_form_->AddBoundaryIntegrator(new mfem::VectorBoundaryLFIntegrator(nat_bc.vectorCoefficient()),
                                     nat_bc.markers());
    }
    l_form_->Assemble();
    rhs_.reset(l_form_->ParallelAssemble());
  } else {
    rhs_  = displacement_.createOnSpace<mfem::HypreParVector>();
    *rhs_ = 0.0;
  }

  // Assemble the stiffness matrix
  K_mat_ = std::unique_ptr<mfem::HypreParMatrix>(K_form_->ParallelAssemble());

  // Eliminate the essential DOFs
  for (auto& bc : bcs_.essentials()) {
    bc.eliminateFromMatrix(*K_mat_);
  }

  // Initialize the eliminate BC RHS vector
  bc_rhs_  = displacement_.createOnSpace<mfem::HypreParVector>();
  *bc_rhs_ = 0.0;

  // Initialize the true vector
  displacement_.initializeTrueVec();
}

void Elasticity::advanceTimestep(double&)
{
  // Initialize the true vector
  displacement_.initializeTrueVec();

  if (is_quasistatic_) {
    QuasiStaticSolve();
  } else {
    SLIC_ERROR_ROOT(mpi_rank_, "Only quasistatics implemented for linear elasticity!");
  }

  // Distribute the shared DOFs
  displacement_.distributeSharedDofs();
  cycle_ += 1;
}

// Solve the Quasi-static system
void Elasticity::QuasiStaticSolve()
{
  // Apply the boundary conditions
  *bc_rhs_ = *rhs_;
  for (auto& bc : bcs_.essentials()) {
    bool should_be_scalar = false;
    bc.apply(*K_mat_, *bc_rhs_, displacement_, time_, should_be_scalar);
  }

  K_inv_.SetOperator(*K_mat_);

  K_inv_.Mult(*bc_rhs_, displacement_.trueVec());
}

Elasticity::~Elasticity() {}

}  // namespace serac

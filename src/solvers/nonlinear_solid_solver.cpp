// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "nonlinear_solid_solver.hpp"

#include "common/logger.hpp"
#include "integrators/hyperelastic_traction_integrator.hpp"
#include "integrators/inc_hyperelastic_integrator.hpp"

namespace serac {

constexpr int NUM_FIELDS = 2;

NonlinearSolidSolver::NonlinearSolidSolver(int order, std::shared_ptr<mfem::ParMesh> pmesh)
    : BaseSolver(pmesh->GetComm(), NUM_FIELDS, order), velocity_(state_[0]), displacement_(state_[1])
{
  velocity_->mesh      = pmesh;
  velocity_->coll      = std::make_shared<mfem::H1_FECollection>(order, pmesh->Dimension());
  velocity_->space     = std::make_shared<mfem::ParFiniteElementSpace>(pmesh.get(), velocity_->coll.get(),
                                                                   pmesh->Dimension(), mfem::Ordering::byVDIM);
  velocity_->gf        = std::make_shared<mfem::ParGridFunction>(velocity_->space.get());
  *velocity_->gf       = 0.0;
  velocity_->true_vec  = std::make_shared<mfem::HypreParVector>(velocity_->space.get());
  *velocity_->true_vec = 0.0;
  velocity_->name      = "velocity";

  displacement_->mesh      = pmesh;
  displacement_->coll      = std::make_shared<mfem::H1_FECollection>(order, pmesh->Dimension());
  displacement_->space     = std::make_shared<mfem::ParFiniteElementSpace>(pmesh.get(), displacement_->coll.get(),
                                                                       pmesh->Dimension(), mfem::Ordering::byVDIM);
  displacement_->gf        = std::make_shared<mfem::ParGridFunction>(displacement_->space.get());
  *displacement_->gf       = 0.0;
  displacement_->true_vec  = std::make_shared<mfem::HypreParVector>(displacement_->space.get());
  *displacement_->true_vec = 0.0;
  displacement_->name      = "displacement";

  // Initialize the mesh node pointers
  reference_nodes_ = std::make_unique<mfem::ParGridFunction>(displacement_->space.get());
  pmesh->GetNodes(*reference_nodes_);
  pmesh->NewNodes(*reference_nodes_);

  deformed_nodes_ = std::make_unique<mfem::ParGridFunction>(*reference_nodes_);

  // Initialize the true DOF vector
  mfem::Array<int> true_offset(NUM_FIELDS + 1);
  int              true_size = velocity_->space->TrueVSize();
  true_offset[0]             = 0;
  true_offset[1]             = true_size;
  true_offset[2]             = 2 * true_size;
  block_                     = std::make_unique<mfem::BlockVector>(true_offset);

  block_->GetBlockView(1, *displacement_->true_vec);
  *displacement_->true_vec = 0.0;

  block_->GetBlockView(0, *velocity_->true_vec);
  *velocity_->true_vec = 0.0;
}

void NonlinearSolidSolver::setDisplacementBCs(const std::set<int>&                     disp_bdr,
                                              std::shared_ptr<mfem::VectorCoefficient> disp_bdr_coef)
{
  setEssentialBCs(disp_bdr, disp_bdr_coef, *displacement_, -1);
}

void NonlinearSolidSolver::setDisplacementBCs(const std::set<int>&               disp_bdr,
                                              std::shared_ptr<mfem::Coefficient> disp_bdr_coef, int component)
{
  setEssentialBCs(disp_bdr, disp_bdr_coef, *displacement_, component);
}

void NonlinearSolidSolver::setTractionBCs(const std::set<int>&                     trac_bdr,
                                          std::shared_ptr<mfem::VectorCoefficient> trac_bdr_coef, int component)
{
  setNaturalBCs(trac_bdr, trac_bdr_coef, component);
}

void NonlinearSolidSolver::setHyperelasticMaterialParameters(const double mu, const double K)
{
  model_.reset(new mfem::NeoHookeanModel(mu, K));
}

void NonlinearSolidSolver::setViscosity(std::unique_ptr<mfem::Coefficient>&& visc_coef)
{
  viscosity_ = std::move(visc_coef);
}

void NonlinearSolidSolver::setDisplacement(mfem::VectorCoefficient& disp_state)
{
  disp_state.SetTime(time_);
  displacement_->gf->ProjectCoefficient(disp_state);
  gf_initialized_[1] = true;
}

void NonlinearSolidSolver::setVelocity(mfem::VectorCoefficient& velo_state)
{
  velo_state.SetTime(time_);
  velocity_->gf->ProjectCoefficient(velo_state);
  gf_initialized_[0] = true;
}

void NonlinearSolidSolver::setSolverParameters(const serac::LinearSolverParameters&    lin_params,
                                               const serac::NonlinearSolverParameters& nonlin_params)
{
  lin_params_    = lin_params;
  nonlin_params_ = nonlin_params;
}

void NonlinearSolidSolver::completeSetup()
{
  // Define the nonlinear form
  auto H_form = std::make_unique<mfem::ParNonlinearForm>(displacement_->space.get());

  // Add the hyperelastic integrator
  if (timestepper_ == serac::TimestepMethod::QuasiStatic) {
    H_form->AddDomainIntegrator(new IncrementalHyperelasticIntegrator(model_.get()));
  } else {
    H_form->AddDomainIntegrator(new mfem::HyperelasticNLFIntegrator(model_.get()));
  }

  // Add the traction integrator
  for (auto& nat_bc_data : nat_bdr_) {
    SLIC_ASSERT_MSG(std::holds_alternative<std::shared_ptr<mfem::VectorCoefficient>>(nat_bc_data.coef),
                    "Traction boundary condition had a non-vector coefficient.");
    H_form->AddBdrFaceIntegrator(nat_bc_data.newVecIntegrator<HyperelasticTractionIntegrator>().release(),
                                 nat_bc_data.markers());
  }

  // Add the essential boundary
  mfem::Array<int> essential_dofs(0);

  // Build the dof array lookup tables
  displacement_->space->BuildDofToArrays();

  // Project the essential boundary coefficients
  for (auto& bc : ess_bdr_) {
    // Project the coefficient
    bc.project(*displacement_->gf, *displacement_->space);

    // Add the vector dofs to the total essential BC dof list
    essential_dofs.Append(bc.getTrueDofs());
  }

  // Remove any duplicates from the essential BC list
  essential_dofs.Sort();
  essential_dofs.Unique();

  H_form->SetEssentialTrueDofs(essential_dofs);

  // The abstract mass bilinear form
  std::unique_ptr<mfem::ParBilinearForm> M_form;

  // The abstract viscosity bilinear form
  std::unique_ptr<mfem::ParBilinearForm> S_form;

  // If dynamic, create the mass and viscosity forms
  if (timestepper_ != serac::TimestepMethod::QuasiStatic) {
    const double              ref_density = 1.0;  // density in the reference configuration
    mfem::ConstantCoefficient rho0(ref_density);

    M_form = std::make_unique<mfem::ParBilinearForm>(displacement_->space.get());

    M_form->AddDomainIntegrator(new mfem::VectorMassIntegrator(rho0));
    M_form->Assemble(0);
    M_form->Finalize(0);

    S_form = std::make_unique<mfem::ParBilinearForm>(displacement_->space.get());
    S_form->AddDomainIntegrator(new mfem::VectorDiffusionIntegrator(*viscosity_));
    S_form->Assemble(0);
    S_form->Finalize(0);
  }

  solver_ = EquationSolver(displacement_->space->GetComm(), lin_params_, nonlin_params_);
  // Set up the jacbian solver based on the linear solver options
  if (lin_params_.prec == serac::Preconditioner::BoomerAMG) {
    SLIC_WARNING_IF(displacement_->space->GetOrdering() == mfem::Ordering::byVDIM,
                    "Attempting to use BoomerAMG with nodal ordering.");
    auto prec_amg = std::make_unique<mfem::HypreBoomerAMG>();
    prec_amg->SetPrintLevel(lin_params_.print_level);
    prec_amg->SetElasticityOptions(displacement_->space.get());
    solver_.SetPreconditioner(std::move(prec_amg));
  } else {
    auto J_hypreSmoother = std::make_unique<mfem::HypreSmoother>();
    J_hypreSmoother->SetType(mfem::HypreSmoother::l1Jacobi);
    J_hypreSmoother->SetPositiveDiagonal(true);
    solver_.SetPreconditioner(std::move(J_hypreSmoother));
  }

  // Set the MFEM abstract operators for use with the internal MFEM solvers
  if (timestepper_ == serac::TimestepMethod::QuasiStatic) {
    solver_.solver().iterative_mode = true;
    nonlinear_oper_                 = std::make_unique<NonlinearSolidQuasiStaticOperator>(std::move(H_form));
    solver_.SetOperator(*nonlinear_oper_);
  } else {
    solver_.solver().iterative_mode = false;
    timedep_oper_                   = std::make_unique<NonlinearSolidDynamicOperator>(
        std::move(H_form), std::move(S_form), std::move(M_form), ess_bdr_, solver_.solver(), lin_params_);
    ode_solver_->Init(*timedep_oper_);
  }
}

// Solve the Quasi-static Newton system
void NonlinearSolidSolver::quasiStaticSolve()
{
  mfem::Vector zero;
  solver_.Mult(zero, *displacement_->true_vec);
}

// Advance the timestep
void NonlinearSolidSolver::advanceTimestep(double& dt)
{
  // Initialize the true vector
  velocity_->gf->GetTrueDofs(*velocity_->true_vec);
  displacement_->gf->GetTrueDofs(*displacement_->true_vec);

  // Set the mesh nodes to the reference configuration
  displacement_->mesh->NewNodes(*reference_nodes_);
  velocity_->mesh->NewNodes(*reference_nodes_);

  if (timestepper_ == serac::TimestepMethod::QuasiStatic) {
    quasiStaticSolve();
  } else {
    ode_solver_->Step(*block_, time_, dt);
  }

  // Distribute the shared DOFs
  velocity_->gf->SetFromTrueDofs(*velocity_->true_vec);
  displacement_->gf->SetFromTrueDofs(*displacement_->true_vec);

  // Update the mesh with the new deformed nodes
  deformed_nodes_->Set(1.0, *displacement_->gf);

  if (timestepper_ == serac::TimestepMethod::QuasiStatic) {
    deformed_nodes_->Add(1.0, *reference_nodes_);
  }

  displacement_->mesh->NewNodes(*deformed_nodes_);
  velocity_->mesh->NewNodes(*deformed_nodes_);

  cycle_ += 1;
}

NonlinearSolidSolver::~NonlinearSolidSolver() {}

}  // namespace serac

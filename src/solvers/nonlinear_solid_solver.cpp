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
    : BaseSolver(pmesh->GetComm(), NUM_FIELDS, order),
      velocity_(state_[0]),
      displacement_(state_[1]),
      newton_solver_(pmesh->GetComm())
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
  setEssentialBCs(disp_bdr, disp_bdr_coef, *displacement_->space, -1);
}

void NonlinearSolidSolver::setDisplacementBCs(const std::set<int>&               disp_bdr,
                                              std::shared_ptr<mfem::Coefficient> disp_bdr_coef, int component)
{
  setEssentialBCs(disp_bdr, disp_bdr_coef, *displacement_->space, component);
}

void NonlinearSolidSolver::setTractionBCs(const std::set<int>&                     trac_bdr,
                                          std::shared_ptr<mfem::VectorCoefficient> trac_bdr_coef, int component)
{
  setNaturalBCs(trac_bdr, trac_bdr_coef, component);
}

void NonlinearSolidSolver::setHyperelasticMaterialParameters(double mu, double K)
{
  model_.reset(new mfem::NeoHookeanModel(mu, K));
}

void NonlinearSolidSolver::setViscosity(std::shared_ptr<mfem::Coefficient> visc) { viscosity_ = visc; }

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
  auto H_form_ = std::make_shared<mfem::ParNonlinearForm>(displacement_->space.get());

  // Add the hyperelastic integrator
  if (timestepper_ == serac::TimestepMethod::QuasiStatic) {
    H_form_->AddDomainIntegrator(new IncrementalHyperelasticIntegrator(model_.get()));
  } else {
    H_form_->AddDomainIntegrator(new mfem::HyperelasticNLFIntegrator(model_.get()));
  }

  // Add the traction integrator
  for (auto& nat_bc_data : nat_bdr_) {
    SLIC_ERROR_IF(!std::holds_alternative<std::shared_ptr<mfem::VectorCoefficient>>(nat_bc_data.coef),
                  "Traction boundary condition had a non-vector coefficient.");
    H_form_->AddBdrFaceIntegrator(
        new HyperelasticTractionIntegrator(*std::get<std::shared_ptr<mfem::VectorCoefficient>>(nat_bc_data.coef)),
        nat_bc_data.markers);
  }

  // Add the essential boundary
  mfem::Array<int> essential_dofs(0);

  // Build the dof array lookup tables
  displacement_->space->BuildDofToArrays();

  // Project the essential boundary coefficients
  for (const auto& bc : ess_bdr_) {
    // Generate the scalar dof list from the vector dof list
    mfem::Array<int> dof_list(bc.true_dofs.Size());
    // Use the const version of the BoundaryCondition for correctness
    std::transform(
        bc.true_dofs.begin(), bc.true_dofs.end(), dof_list.begin(), [&bc = std::as_const(bc), this](const int tdof) {
          auto dof = displacement_->space->VDofToDof(tdof);
          SLIC_WARNING_IF((bc.component != -1) && (tdof != displacement_->space->DofToVDof(dof, bc.component)),
                          "Single-component boundary condition tdofs do not match provided component.");
          return dof;
        });

    // Project the coefficient
    if (bc.component == -1) {
      // If it contains all components, project the vector
      SLIC_ERROR_IF(!std::holds_alternative<std::shared_ptr<mfem::VectorCoefficient>>(bc.coef),
                    "Displacement boundary condition contained all components but had a non-vector coefficient.");
      displacement_->gf->ProjectCoefficient(*std::get<std::shared_ptr<mfem::VectorCoefficient>>(bc.coef), dof_list);
    } else {
      // If it is only a single component, project the scalar
      SLIC_ERROR_IF(!std::holds_alternative<std::shared_ptr<mfem::Coefficient>>(bc.coef),
                    "Displacement boundary condition contained a single component but had a non-scalar coefficient.");
      displacement_->gf->ProjectCoefficient(*std::get<std::shared_ptr<mfem::Coefficient>>(bc.coef), dof_list,
                                            bc.component);
    }

    // Add the vector dofs to the total essential BC dof list
    essential_dofs.Append(bc.true_dofs);
  }

  // Remove any duplicates from the essential BC list
  essential_dofs.Sort();
  essential_dofs.Unique();

  H_form_->SetEssentialTrueDofs(essential_dofs);

  // The abstract mass bilinear form
  std::shared_ptr<mfem::ParBilinearForm> M_form_;

  // The abstract viscosity bilinear form
  std::shared_ptr<mfem::ParBilinearForm> S_form_;

  // If dynamic, create the mass and viscosity forms
  if (timestepper_ != serac::TimestepMethod::QuasiStatic) {
    const double              ref_density = 1.0;  // density in the reference configuration
    mfem::ConstantCoefficient rho0(ref_density);

    M_form_ = std::make_shared<mfem::ParBilinearForm>(displacement_->space.get());

    M_form_->AddDomainIntegrator(new mfem::VectorMassIntegrator(rho0));
    M_form_->Assemble(0);
    M_form_->Finalize(0);

    S_form_ = std::make_shared<mfem::ParBilinearForm>(displacement_->space.get());
    S_form_->AddDomainIntegrator(new mfem::VectorDiffusionIntegrator(*viscosity_));
    S_form_->Assemble(0);
    S_form_->Finalize(0);
  }

  // Set up the jacbian solver based on the linear solver options
  std::unique_ptr<mfem::IterativeSolver> iter_solver;

  if (lin_params_.prec == serac::Preconditioner::BoomerAMG) {
    SLIC_WARNING_IF(displacement_->space->GetOrdering() == mfem::Ordering::byVDIM,
                    "Attempting to use BoomerAMG with nodal ordering.");
    auto prec_amg = std::make_unique<mfem::HypreBoomerAMG>();
    prec_amg->SetPrintLevel(lin_params_.print_level);
    prec_amg->SetElasticityOptions(displacement_->space.get());
    J_prec_ = std::move(prec_amg);

    iter_solver = std::make_unique<mfem::GMRESSolver>(displacement_->space->GetComm());
  } else {
    auto J_hypreSmoother = std::make_unique<mfem::HypreSmoother>();
    J_hypreSmoother->SetType(mfem::HypreSmoother::l1Jacobi);
    J_hypreSmoother->SetPositiveDiagonal(true);
    J_prec_ = std::move(J_hypreSmoother);

    iter_solver = std::make_unique<mfem::MINRESSolver>(displacement_->space->GetComm());
  }

  iter_solver->SetRelTol(lin_params_.rel_tol);
  iter_solver->SetAbsTol(lin_params_.abs_tol);
  iter_solver->SetMaxIter(lin_params_.max_iter);
  iter_solver->SetPrintLevel(lin_params_.print_level);
  iter_solver->SetPreconditioner(*J_prec_);
  J_solver_ = std::move(iter_solver);

  // Set the newton solve parameters
  newton_solver_.SetSolver(*J_solver_);
  newton_solver_.SetPrintLevel(nonlin_params_.print_level);
  newton_solver_.SetRelTol(nonlin_params_.rel_tol);
  newton_solver_.SetAbsTol(nonlin_params_.abs_tol);
  newton_solver_.SetMaxIter(nonlin_params_.max_iter);

  // Set the MFEM abstract operators for use with the internal MFEM solvers
  if (timestepper_ == serac::TimestepMethod::QuasiStatic) {
    newton_solver_.iterative_mode = true;
    nonlinear_oper_               = std::make_shared<NonlinearSolidQuasiStaticOperator>(H_form_);
    newton_solver_.SetOperator(*nonlinear_oper_);
  } else {
    newton_solver_.iterative_mode = false;
    timedep_oper_ = std::make_shared<NonlinearSolidDynamicOperator>(H_form_, S_form_, M_form_, ess_bdr_, newton_solver_,
                                                                    lin_params_);
    ode_solver_->Init(*timedep_oper_);
  }
}

// Solve the Quasi-static Newton system
void NonlinearSolidSolver::QuasiStaticSolve()
{
  mfem::Vector zero;
  newton_solver_.Mult(zero, *displacement_->true_vec);
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
    QuasiStaticSolve();
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

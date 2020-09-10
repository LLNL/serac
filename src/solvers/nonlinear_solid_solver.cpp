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

NonlinearSolidSolver::NonlinearSolidSolver(int order, std::shared_ptr<mfem::ParMesh> mesh)
    : BaseSolver(mesh, NUM_FIELDS, order),
      velocity_(std::make_shared<FiniteElementState>(*mesh, FEStateOptions{.order = order, .name = "velocity"})),
      displacement_(std::make_shared<FiniteElementState>(*mesh, FEStateOptions{.order = order, .name = "displacement"}))
{
  state_[0] = velocity_;
  state_[1] = displacement_;

  // Initialize the mesh node pointers
  reference_nodes_ = displacement_->createOnSpace<mfem::ParGridFunction>();
  mesh->GetNodes(*reference_nodes_);
  mesh->NewNodes(*reference_nodes_);

  deformed_nodes_ = std::make_unique<mfem::ParGridFunction>(*reference_nodes_);

  // Initialize the true DOF vector
  mfem::Array<int> true_offset(NUM_FIELDS + 1);
  int              true_size = velocity_->space().TrueVSize();
  true_offset[0]             = 0;
  true_offset[1]             = true_size;
  true_offset[2]             = 2 * true_size;
  block_                     = std::make_unique<mfem::BlockVector>(true_offset);

  block_->GetBlockView(1, displacement_->trueVec());
  displacement_->trueVec() = 0.0;

  block_->GetBlockView(0, velocity_->trueVec());
  velocity_->trueVec() = 0.0;
}

void NonlinearSolidSolver::setDisplacementBCs(const std::set<int>&                     disp_bdr,
                                              std::shared_ptr<mfem::VectorCoefficient> disp_bdr_coef)
{
  bcs_.addEssential(disp_bdr, disp_bdr_coef, *displacement_, -1);
}

void NonlinearSolidSolver::setDisplacementBCs(const std::set<int>&               disp_bdr,
                                              std::shared_ptr<mfem::Coefficient> disp_bdr_coef, int component)
{
  bcs_.addEssential(disp_bdr, disp_bdr_coef, *displacement_, component);
}

void NonlinearSolidSolver::setTractionBCs(const std::set<int>&                     trac_bdr,
                                          std::shared_ptr<mfem::VectorCoefficient> trac_bdr_coef, int component)
{
  bcs_.addNatural(trac_bdr, trac_bdr_coef, component);
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
  displacement_->project(disp_state);
  gf_initialized_[1] = true;
}

void NonlinearSolidSolver::setVelocity(mfem::VectorCoefficient& velo_state)
{
  velo_state.SetTime(time_);
  velocity_->project(velo_state);
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
  auto H_form = displacement_->createOnSpace<mfem::ParNonlinearForm>();

  // Add the hyperelastic integrator
  if (timestepper_ == serac::TimestepMethod::QuasiStatic) {
    H_form->AddDomainIntegrator(new IncrementalHyperelasticIntegrator(model_.get()));
  } else {
    H_form->AddDomainIntegrator(new mfem::HyperelasticNLFIntegrator(model_.get()));
  }

  // Add the traction integrator
  for (auto& nat_bc_data : bcs_.naturals()) {
    H_form->AddBdrFaceIntegrator(new HyperelasticTractionIntegrator(nat_bc_data.vectorCoefficient()),
                                 nat_bc_data.markers());
  }

  // Build the dof array lookup tables
  displacement_->space().BuildDofToArrays();

  // Project the essential boundary coefficients
  for (auto& bc : bcs_.essentials()) {
    // Project the coefficient
    bc.project(*displacement_);
  }

  // The abstract mass bilinear form
  std::unique_ptr<mfem::ParBilinearForm> M_form;

  // The abstract viscosity bilinear form
  std::unique_ptr<mfem::ParBilinearForm> S_form;

  // If dynamic, create the mass and viscosity forms
  if (timestepper_ != serac::TimestepMethod::QuasiStatic) {
    const double              ref_density = 1.0;  // density in the reference configuration
    mfem::ConstantCoefficient rho0(ref_density);

    M_form = displacement_->createOnSpace<mfem::ParBilinearForm>();

    M_form->AddDomainIntegrator(new mfem::VectorMassIntegrator(rho0));
    M_form->Assemble(0);
    M_form->Finalize(0);

    S_form = displacement_->createOnSpace<mfem::ParBilinearForm>();
    S_form->AddDomainIntegrator(new mfem::VectorDiffusionIntegrator(*viscosity_));
    S_form->Assemble(0);
    S_form->Finalize(0);
  }

  solver_ = EquationSolver(displacement_->comm(), lin_params_, nonlin_params_);
  // Set up the jacbian solver based on the linear solver options
  if (lin_params_.prec == serac::Preconditioner::BoomerAMG) {
    SLIC_WARNING_IF(displacement_->space().GetOrdering() == mfem::Ordering::byVDIM,
                    "Attempting to use BoomerAMG with nodal ordering.");
    auto prec_amg = std::make_unique<mfem::HypreBoomerAMG>();
    prec_amg->SetPrintLevel(lin_params_.print_level);
    prec_amg->SetElasticityOptions(&displacement_->space());
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
    nonlinear_oper_                 = std::make_unique<NonlinearSolidQuasiStaticOperator>(std::move(H_form), bcs_);
    solver_.SetOperator(*nonlinear_oper_);
  } else {
    solver_.solver().iterative_mode = false;
    timedep_oper_                   = std::make_unique<NonlinearSolidDynamicOperator>(
        std::move(H_form), std::move(S_form), std::move(M_form), bcs_, solver_.solver(), lin_params_);
    ode_solver_->Init(*timedep_oper_);
  }
}

// Solve the Quasi-static Newton system
void NonlinearSolidSolver::quasiStaticSolve()
{
  mfem::Vector zero;
  solver_.Mult(zero, displacement_->trueVec());
}

// Advance the timestep
void NonlinearSolidSolver::advanceTimestep(double& dt)
{
  // Initialize the true vector
  velocity_->initializeTrueVec();
  displacement_->initializeTrueVec();

  // Set the mesh nodes to the reference configuration
  mesh_->NewNodes(*reference_nodes_);

  if (timestepper_ == serac::TimestepMethod::QuasiStatic) {
    quasiStaticSolve();
  } else {
    ode_solver_->Step(*block_, time_, dt);
  }

  // Distribute the shared DOFs
  velocity_->distributeSharedDofs();
  displacement_->distributeSharedDofs();

  // Update the mesh with the new deformed nodes
  deformed_nodes_->Set(1.0, displacement_->gridFunc());

  if (timestepper_ == serac::TimestepMethod::QuasiStatic) {
    deformed_nodes_->Add(1.0, *reference_nodes_);
  }

  mesh_->NewNodes(*deformed_nodes_);

  cycle_ += 1;
}

NonlinearSolidSolver::~NonlinearSolidSolver() {}

}  // namespace serac

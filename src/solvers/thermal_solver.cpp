// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "thermal_solver.hpp"

#include "common/logger.hpp"

namespace serac {

constexpr int NUM_FIELDS = 1;

ThermalSolver::ThermalSolver(int order, std::shared_ptr<mfem::ParMesh> pmesh)
    : BaseSolver(pmesh->GetComm(), NUM_FIELDS, order),
      temperature_(std::make_shared<FiniteElementState>(
          pmesh,
          FEStateOptions{.order = order, .space_dim = 1, .ordering = mfem::Ordering::byNODES, .name = "temperature"}))
{
  state_[0] = temperature_;
}

void ThermalSolver::setTemperature(mfem::Coefficient& temp)
{
  // Project the coefficient onto the grid function
  temp.SetTime(time_);
  temperature_->project(temp);
  gf_initialized_[0] = true;
}

void ThermalSolver::setTemperatureBCs(const std::set<int>& ess_bdr, std::shared_ptr<mfem::Coefficient> ess_bdr_coef)
{
  setEssentialBCs(ess_bdr, ess_bdr_coef, *temperature_);
}

void ThermalSolver::setFluxBCs(const std::set<int>& nat_bdr, std::shared_ptr<mfem::Coefficient> nat_bdr_coef)
{
  // Set the natural (integral) boundary condition
  setNaturalBCs(nat_bdr, nat_bdr_coef);
}

void ThermalSolver::setConductivity(std::shared_ptr<mfem::Coefficient> kappa)
{
  // Set the conduction coefficient
  kappa_ = kappa;
}

void ThermalSolver::setSource(std::shared_ptr<mfem::Coefficient> source)
{
  // Set the body source integral coefficient
  source_ = source;
}

void ThermalSolver::setLinearSolverParameters(const serac::LinearSolverParameters& params)
{
  // Save the solver params object
  // TODO: separate the M and K solver params
  lin_params_ = params;
}

void ThermalSolver::completeSetup()
{
  SLIC_ASSERT_MSG(kappa_ != nullptr, "Conductivity not set in ThermalSolver!");

  // Add the domain diffusion integrator to the K form and assemble the matrix
  K_form_ = temperature_->createTensorOnSpace<mfem::ParBilinearForm>();
  K_form_->AddDomainIntegrator(new mfem::DiffusionIntegrator(*kappa_));
  K_form_->Assemble(0);  // keep sparsity pattern of M and K the same
  K_form_->Finalize();

  // Add the body source to the RS if specified
  l_form_ = temperature_->createTensorOnSpace<mfem::ParLinearForm>();
  if (source_ != nullptr) {
    l_form_->AddDomainIntegrator(new mfem::DomainLFIntegrator(*source_));
    rhs_.reset(l_form_->ParallelAssemble());
  } else {
    rhs_  = temperature_->createTensorOnSpace<mfem::HypreParVector>();
    *rhs_ = 0.0;
  }

  // Assemble the stiffness matrix
  K_mat_.reset(K_form_->ParallelAssemble());

  // Eliminate the essential DOFs from the stiffness matrix
  for (auto& bc : ess_bdr_) {
    bc.eliminateFromMatrix(*K_mat_);
  }

  // Initialize the eliminated BC RHS vector
  bc_rhs_  = temperature_->createTensorOnSpace<mfem::HypreParVector>();
  *bc_rhs_ = 0.0;

  // Initialize the true vector
  temperature_->initializeTrueVec();

  if (timestepper_ != serac::TimestepMethod::QuasiStatic) {
    // If dynamic, assemble the mass matrix
    M_form_ = temperature_->createTensorOnSpace<mfem::ParBilinearForm>();
    M_form_->AddDomainIntegrator(new mfem::MassIntegrator());
    M_form_->Assemble(0);  // keep sparsity pattern of M and K the same
    M_form_->Finalize();

    M_mat_.reset(M_form_->ParallelAssemble());

    // Make the time integration operator and set the appropriate matricies
    dyn_oper_ = std::make_unique<DynamicConductionOperator>(temperature_->space(), lin_params_, ess_bdr_);
    dyn_oper_->setMatrices(M_mat_, K_mat_);
    dyn_oper_->setLoadVector(rhs_);

    ode_solver_->Init(*dyn_oper_);
  }
}

void ThermalSolver::quasiStaticSolve()
{
  // Apply the boundary conditions
  *bc_rhs_ = *rhs_;
  for (auto& bc : ess_bdr_) {
    bc.apply(*K_mat_, *bc_rhs_, *temperature_, time_);
  }

  // Solve the stiffness using CG with Jacobi preconditioning
  // and the given solverparams
  solver_ = AlgebraicSolver(temperature_->comm(), lin_params_);

  auto hypre_smoother = std::make_unique<mfem::HypreSmoother>();
  hypre_smoother->SetType(mfem::HypreSmoother::Jacobi);

  solver_.SetPreconditioner(std::move(hypre_smoother));

  solver_.solver().iterative_mode = false;
  solver_.SetOperator(*K_mat_);

  // Perform the linear solve
  solver_.Mult(*bc_rhs_, temperature_->trueVec());
}

void ThermalSolver::advanceTimestep(double& dt)
{
  // Initialize the true vector
  temperature_->initializeTrueVec();

  if (timestepper_ == serac::TimestepMethod::QuasiStatic) {
    quasiStaticSolve();
  } else {
    SLIC_ASSERT_MSG(gf_initialized_[0], "Thermal state not initialized!");

    // Step the time integrator
    ode_solver_->Step(temperature_->trueVec(), time_, dt);
  }

  // Distribute the shared DOFs
  temperature_->distributeSharedDofs();
  cycle_ += 1;
}

}  // namespace serac

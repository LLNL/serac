// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "physics/thermal_solver_rework.hpp"

#include "infrastructure/logger.hpp"

namespace serac {

constexpr int NUM_FIELDS = 1;

ThermalSolverRework::ThermalSolverRework(int order, std::shared_ptr<mfem::ParMesh> mesh)
    : BaseSolver(mesh, NUM_FIELDS, order),
      temperature_(std::make_shared<FiniteElementState>(
          *mesh,
          FEStateOptions{.order = order, .space_dim = 1, .ordering = mfem::Ordering::byNODES, .name = "temperature"}))
{
  state_[0] = temperature_;
}

void ThermalSolverRework::setTemperature(mfem::Coefficient& temp)
{
  // Project the coefficient onto the grid function
  temp.SetTime(time_);
  temperature_->project(temp);
  gf_initialized_[0] = true;
}

void ThermalSolverRework::setTemperatureBCs(const std::set<int>&               temp_bdr,
                                            std::shared_ptr<mfem::Coefficient> temp_bdr_coef)
{
  bcs_.addEssential(temp_bdr, temp_bdr_coef, *temperature_);
}

void ThermalSolverRework::setFluxBCs(const std::set<int>& flux_bdr, std::shared_ptr<mfem::Coefficient> flux_bdr_coef)
{
  // Set the natural (integral) boundary condition
  bcs_.addNatural(flux_bdr, flux_bdr_coef, -1);
}

void ThermalSolverRework::setConductivity(std::unique_ptr<mfem::Coefficient>&& kappa)
{
  // Set the conduction coefficient
  kappa_ = std::move(kappa);
}

void ThermalSolverRework::setSource(std::unique_ptr<mfem::Coefficient>&& source)
{
  // Set the body source integral coefficient
  source_ = std::move(source);
}

void ThermalSolverRework::setLinearSolverParameters(const serac::LinearSolverParameters& params)
{
  // Save the solver params object
  // TODO: separate the M and K solver params
  lin_params_ = params;
}

void ThermalSolverRework::completeSetup()
{
  SLIC_ASSERT_MSG(kappa_ != nullptr, "Conductivity not set in ThermalSolverRework!");

  // Add the domain diffusion integrator to the K form and assemble the matrix
  K_form_ = temperature_->createOnSpace<mfem::ParBilinearForm>();
  K_form_->AddDomainIntegrator(new mfem::DiffusionIntegrator(*kappa_));
  K_form_->Assemble(0);  // keep sparsity pattern of M and K the same
  K_form_->Finalize();

  K_.reset(K_form_->ParallelAssemble());

  // Add the body source to the RS if specified
  l_form_ = temperature_->createOnSpace<mfem::ParLinearForm>();
  if (source_ != nullptr) {
    l_form_->AddDomainIntegrator(new mfem::DomainLFIntegrator(*source_));
    rhs_.reset(l_form_->ParallelAssemble());
  } else {
    rhs_  = temperature_->createOnSpace<mfem::HypreParVector>();
    *rhs_ = 0.0;
  }

  // Initialize the eliminated BC RHS vector
  bc_rhs_  = temperature_->createOnSpace<mfem::HypreParVector>();
  *bc_rhs_ = 0.0;

  // Initialize the true vector
  temperature_->initializeTrueVec();

  if (timestepper_ == serac::TimestepMethod::QuasiStatic) {
    // Eliminate the essential DOFs from the stiffness matrix
    for (auto& bc : bcs_.essentials()) {
      bc.eliminateFromMatrix(*K_);
    }

  } else {
    // If dynamic, assemble the mass matrix
    M_form_ = temperature_->createOnSpace<mfem::ParBilinearForm>();
    M_form_->AddDomainIntegrator(new mfem::MassIntegrator());
    M_form_->Assemble(0);  // keep sparsity pattern of M and K the same
    M_form_->Finalize();

    M_.reset(M_form_->ParallelAssemble());

    auto preconditioner = std::make_unique<mfem::HypreSmoother>();
    preconditioner->SetType(mfem::HypreSmoother::Jacobi);

    invT_ = EquationSolver(M_->GetComm(), lin_params_);
    invT_.linearSolver().iterative_mode = false;
    invT_.SetPreconditioner(std::move(preconditioner));

    uc       = temperature_->trueVec();
    uc_plus  = temperature_->trueVec();
    uc_minus = temperature_->trueVec();
    duc_dt   = temperature_->trueVec();

    double epsilon = 1.0e-8;

    ode_ = FirstOrderODE(temperature_->trueVec().Size(),
      [=, previous_dt = -1.0] (const double t, const double dt, const mfem::Vector& u, mfem::Vector& du_dt) mutable {

      if (dt != previous_dt) {
        // T = M + dt K
        T_.reset(mfem::Add(1.0, *M_, dt, *K_));

        // Eliminate the essential DOFs from the T matrix
        bcs_.eliminateAllEssentialDofsFromMatrix(*T_);
        invT_.SetOperator(*T_);

        previous_dt = dt;
      }

      uf = u;
      uf.SetSubVector(bcs_.allEssentialDofs(), 0.0);

      uc       = 0.0;
      uc_plus  = 0.0;
      uc_minus = 0.0;
      for (const auto& bc : bcs_.essentials()) {
        bc.projectBdrToDofs(uc, t);
        bc.projectBdrToDofs(uc_plus, t + epsilon);
        bc.projectBdrToDofs(uc_minus, t - epsilon);
      }
      duc_dt = (uc_plus - uc_minus) / (2.0 * epsilon);

      du_dt = invT_ * (*rhs_ - *M_ * duc_dt - *K_ * uc - *K_ * uf);
    });

    ode_solver_->Init(ode_);
  }
}

void ThermalSolverRework::quasiStaticSolve()
{
  // Apply the boundary conditions
  *bc_rhs_ = *rhs_;
  for (auto& bc : bcs_.essentials()) {
    bc.apply(*K_, *bc_rhs_, *temperature_, time_);
  }

  // Solve the stiffness using CG with Jacobi preconditioning
  // and the given solverparams
  solver_ = EquationSolver(temperature_->comm(), lin_params_);

  auto hypre_smoother = std::make_unique<mfem::HypreSmoother>();
  hypre_smoother->SetType(mfem::HypreSmoother::Jacobi);

  solver_.SetPreconditioner(std::move(hypre_smoother));

  solver_.linearSolver().iterative_mode = false;
  solver_.SetOperator(*K_);

  // Perform the linear solve
  solver_.Mult(*bc_rhs_, temperature_->trueVec());
}

void ThermalSolverRework::advanceTimestep(double& dt)
{
  // Initialize the true vector
  temperature_->initializeTrueVec();

  if (timestepper_ == serac::TimestepMethod::QuasiStatic) {
    quasiStaticSolve();
  } else {
    SLIC_ASSERT_MSG(gf_initialized_[0], "Thermal state not initialized!");

    // integrate forward in time
    ode_solver_->Step(temperature_->trueVec(), time_, dt);

    //
    for (const auto& bc : bcs_.essentials()) {
      bc.projectBdrToDofs(temperature_->trueVec(), time_);
    }
  }

  // Distribute the shared DOFs
  temperature_->distributeSharedDofs();
  cycle_ += 1;
}

}  // namespace serac

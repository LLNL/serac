// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "thermal_solver.hpp"

#include "common/logger.hpp"

namespace serac {

constexpr int NUM_FIELDS = 1;

ThermalSolver::ThermalSolver(int order, std::shared_ptr<mfem::ParMesh> mesh)
    : BaseSolver(mesh, NUM_FIELDS, order),
      temperature_(std::make_shared<FiniteElementState>(
          *mesh,
          FEStateOptions{.order = order, .space_dim = 1, .ordering = mfem::Ordering::byNODES, .name = "temperature"})),
      ode(temperature_->space().GetTrueVSize())
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

void ThermalSolver::setTemperatureBCs(const std::set<int>& temp_bdr, std::shared_ptr<mfem::Coefficient> temp_bdr_coef)
{
  bcs_.addEssential(temp_bdr, temp_bdr_coef, *temperature_);
}

void ThermalSolver::setFluxBCs(const std::set<int>& flux_bdr, std::shared_ptr<mfem::Coefficient> flux_bdr_coef)
{
  // Set the natural (integral) boundary condition
  bcs_.addNatural(flux_bdr, flux_bdr_coef, -1);
}

void ThermalSolver::setConductivity(std::unique_ptr<mfem::Coefficient>&& kappa)
{
  // Set the conduction coefficient
  kappa_ = std::move(kappa);
}

void ThermalSolver::setSource(std::unique_ptr<mfem::Coefficient>&& source)
{
  // Set the body source integral coefficient
  source_ = std::move(source);
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
  K_form_ = temperature_->createOnSpace<mfem::ParBilinearForm>();
  K_form_->AddDomainIntegrator(new mfem::DiffusionIntegrator(*kappa_));
  K_form_->Assemble(0);  // keep sparsity pattern of M and K the same
  K_form_->Finalize();

  // Add the body source to the RS if specified
  l_form_ = temperature_->createOnSpace<mfem::ParLinearForm>();
  if (source_ != nullptr) {
    l_form_->AddDomainIntegrator(new mfem::DomainLFIntegrator(*source_));
    rhs_.reset(l_form_->ParallelAssemble());
  } else {
    rhs_  = temperature_->createOnSpace<mfem::HypreParVector>();
    *rhs_ = 0.0;
  }

  // Assemble the stiffness matrix
  K_mat_.reset(K_form_->ParallelAssemble());

  // Eliminate the essential DOFs from the stiffness matrix
  for (auto& bc : bcs_.essentials()) {
    bc.eliminateFromMatrix(*K_mat_);
  }

  // Initialize the eliminated BC RHS vector
  bc_rhs_  = temperature_->createOnSpace<mfem::HypreParVector>();
  *bc_rhs_ = 0.0;

  // Initialize the true vector
  temperature_->initializeTrueVec();

  if (timestepper_ != serac::TimestepMethod::QuasiStatic) {
    // If dynamic, assemble the mass matrix
    M_form_ = temperature_->createOnSpace<mfem::ParBilinearForm>();
    M_form_->AddDomainIntegrator(new mfem::MassIntegrator());
    M_form_->Assemble(0);  // keep sparsity pattern of M and K the same
    M_form_->Finalize();

    M_mat_.reset(M_form_->ParallelAssemble());

    // Make the time integration operator and set the appropriate matricies
    #if 0
    ode.explicit_func = []() {

    };

    ode.implicit_func = []() {

    };

    ode.explicit_func = [&](const double t, const mfem::Vector& u, mfem::Vector& du_dt) {
      mfem::Vector uf = u;
      uf[1]           = 0;
      du_dt           = invH * (fext(t) - K * uf - K * uc(t) - M * duc_dt(t));
    };
    ode.implicit_func = [&, dt_prev = -1.0, invT = mfem::DenseMatrix(3, 3)](
                            const double t, const double dt, const mfem::Vector& u, mfem::Vector& du_dt) mutable {
      if (dt != dt_prev) {
        // T = M + dt K
        T.reset(mfem::Add(1.0, *M_, dt, *K_));

        // Eliminate the off-diagonal entries associated with essential DOFs
        for (auto& bc : ess_bdr_) {
          delete T->EliminateRowsCols(bc.getTrueDofs()));
        }
        inv_T.SetOperator(*T);
      }

      uc = 0;
      duc_dt = 0;
      for (auto& bc : ess_bdr_) {
        bc.projectBdr(*state_gf_, t);
        state_gf_->SetFromTrueDofs(uc);
        state_gf_->GetTrueDofs(uc);
      }
      uf = u - uc;

      mfem::Vector f = fext(t) - M * duc_dt(t) - K * uc(t) - K * uf;
      f[1]           = 0;

      du_dt = invT * f;

      dt_prev = dt;

    };
    #endif

    dyn_oper_ = std::make_unique<DynamicConductionOperator>(temperature_->space(), lin_params_, bcs_);
    dyn_oper_->setMatrices(M_mat_.get(), K_mat_.get());
    dyn_oper_->setLoadVector(rhs_.get());

    ode_solver_->Init(*dyn_oper_);
  }
}

void ThermalSolver::quasiStaticSolve()
{
  // Apply the boundary conditions
  *bc_rhs_ = *rhs_;
  for (auto& bc : bcs_.essentials()) {
    bc.apply(*K_mat_, *bc_rhs_, *temperature_, time_);
  }

  // Solve the stiffness using CG with Jacobi preconditioning
  // and the given solverparams
  solver_ = EquationSolver(temperature_->comm(), lin_params_);

  auto hypre_smoother = std::make_unique<mfem::HypreSmoother>();
  hypre_smoother->SetType(mfem::HypreSmoother::Jacobi);

  solver_.SetPreconditioner(std::move(hypre_smoother));

  solver_.linearSolver().iterative_mode = false;
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

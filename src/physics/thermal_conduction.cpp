// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "physics/thermal_conduction.hpp"

#include "infrastructure/logger.hpp"

namespace serac {

constexpr int NUM_FIELDS = 1;

ThermalConduction::ThermalConduction(int order, std::shared_ptr<mfem::ParMesh> mesh,
                                     const ThermalConductionParameters& params)
    : BasePhysics(mesh, NUM_FIELDS, order),
      temperature_(std::make_shared<FiniteElementState>(
          *mesh,
          FEStateOptions{.order = order, .space_dim = 1, .ordering = mfem::Ordering::byNODES, .name = "temperature"}))
{
  state_[0] = temperature_;
  std::visit(
      [this, &mesh](const auto& config) {
        // Just the K solver - quasistatic
        if constexpr (std::is_same_v<std::decay_t<decltype(config)>, LinearSolverParameters>) {
          setTimestepper(TimestepMethod::QuasiStatic);
          K_inv_ = EquationSolver(mesh->GetComm(), config);
        }
        // Otherwise - dynamic with M and T configs
        else {
          setTimestepper(std::get<0>(config));
          dyn_oper_params_ = std::make_pair(std::get<1>(config), std::get<2>(config));
        }
      },
      params);
}

void ThermalConduction::setTemperature(mfem::Coefficient& temp)
{
  // Project the coefficient onto the grid function
  temp.SetTime(time_);
  temperature_->project(temp);
  gf_initialized_[0] = true;
}

void ThermalConduction::setTemperatureBCs(const std::set<int>&               temp_bdr,
                                          std::shared_ptr<mfem::Coefficient> temp_bdr_coef)
{
  bcs_.addEssential(temp_bdr, temp_bdr_coef, *temperature_);
}

void ThermalConduction::setFluxBCs(const std::set<int>& flux_bdr, std::shared_ptr<mfem::Coefficient> flux_bdr_coef)
{
  // Set the natural (integral) boundary condition
  bcs_.addNatural(flux_bdr, flux_bdr_coef, -1);
}

void ThermalConduction::setConductivity(std::unique_ptr<mfem::Coefficient>&& kappa)
{
  // Set the conduction coefficient
  kappa_ = std::move(kappa);
}

void ThermalConduction::setSource(std::unique_ptr<mfem::Coefficient>&& source)
{
  // Set the body source integral coefficient
  source_ = std::move(source);
}

void ThermalConduction::completeSetup()
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

    // Make the time integration operator and set the appropriate matrices
    dyn_oper_ = std::make_unique<DynamicConductionOperator>(temperature_->space(), dyn_oper_params_->first,
                                                            dyn_oper_params_->second, bcs_);
    dyn_oper_->setMatrices(M_mat_.get(), K_mat_.get());
    dyn_oper_->setLoadVector(rhs_.get());

    ode_solver_->Init(*dyn_oper_);
  }
}

void ThermalConduction::quasiStaticSolve()
{
  // Apply the boundary conditions
  *bc_rhs_ = *rhs_;
  for (auto& bc : bcs_.essentials()) {
    bc.apply(*K_mat_, *bc_rhs_, *temperature_, time_);
  }

  // Solve the stiffness using CG with Jacobi preconditioning
  // and the given solverparams
  // auto hypre_smoother = std::make_unique<mfem::HypreSmoother>();
  // hypre_smoother->SetType(mfem::HypreSmoother::Jacobi);

  // solver_.SetPreconditioner(std::move(hypre_smoother));

  K_inv_->linearSolver().iterative_mode = false;
  K_inv_->SetOperator(*K_mat_);

  // Perform the linear solve
  K_inv_->Mult(*bc_rhs_, temperature_->trueVec());
}

void ThermalConduction::advanceTimestep(double& dt)
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

// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/thermal_conduction.hpp"

#include "serac/infrastructure/logger.hpp"
#include "serac/numerics/expr_template_ops.hpp"

namespace serac {

constexpr int NUM_FIELDS = 1;

ThermalConduction::ThermalConduction(int order, std::shared_ptr<mfem::ParMesh> mesh, const SolverOptions& options)
    : BasePhysics(mesh, NUM_FIELDS, order),
      temperature_(StateManager::newState(
          *mesh,
          FiniteElementState::Options{
              .order = order, .space_dim = 1, .ordering = mfem::Ordering::byNODES, .name = "temperature"})),
      residual_(temperature_.space().TrueVSize()),
      ode_(temperature_.space().TrueVSize(), {.u = u_, .dt = dt_, .du_dt = previous_, .previous_dt = previous_dt_},
           nonlin_solver_, bcs_)
{
  state_.push_back(temperature_);

  nonlin_solver_ = mfem_ext::EquationSolver(mesh->GetComm(), options.T_lin_options, options.T_nonlin_options);
  nonlin_solver_.SetOperator(residual_);

  // Check for dynamic mode
  if (options.dyn_options) {
    ode_.SetTimestepper(options.dyn_options->timestepper);
    ode_.SetEnforcementMethod(options.dyn_options->enforcement_method);
    is_quasistatic_ = false;
  } else {
    is_quasistatic_ = true;
  }

  dt_          = 0.0;
  previous_dt_ = -1.0;

  int true_size = temperature_.space().TrueVSize();
  u_.SetSize(true_size);
  previous_.SetSize(true_size);
  previous_ = 0.0;

  zero_.SetSize(true_size);
  zero_ = 0.0;

  // Default to constant value of 1.0 for density and specific heat capacity
  cp_  = std::make_unique<mfem::ConstantCoefficient>(1.0);
  rho_ = std::make_unique<mfem::ConstantCoefficient>(1.0);
}

void ThermalConduction::setTemperature(mfem::Coefficient& temp)
{
  // Project the coefficient onto the grid function
  temp.SetTime(time_);
  temperature_.project(temp);
  gf_initialized_[0] = true;
}

void ThermalConduction::setTemperatureBCs(const std::set<int>&               temp_bdr,
                                          std::shared_ptr<mfem::Coefficient> temp_bdr_coef)
{
  bcs_.addEssential(temp_bdr, temp_bdr_coef, temperature_);
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

void ThermalConduction::setSpecificHeatCapacity(std::unique_ptr<mfem::Coefficient>&& cp)
{
  // Set the specific heat capacity coefficient
  cp_ = std::move(cp);
}

void ThermalConduction::setDensity(std::unique_ptr<mfem::Coefficient>&& rho)
{
  // Set the density coefficient
  rho_ = std::move(rho);
}

void ThermalConduction::completeSetup()
{
  SLIC_ASSERT_MSG(kappa_, "Conductivity not set in ThermalSolver!");

  // Add the domain diffusion integrator to the K form and assemble the matrix
  K_form_ = temperature_.createOnSpace<mfem::ParBilinearForm>();
  K_form_->AddDomainIntegrator(new mfem::DiffusionIntegrator(*kappa_));
  K_form_->Assemble(0);  // keep sparsity pattern of M and K the same
  K_form_->Finalize();

  // Add the body source to the RS if specified
  l_form_ = temperature_.createOnSpace<mfem::ParLinearForm>();
  if (source_) {
    l_form_->AddDomainIntegrator(new mfem::DomainLFIntegrator(*source_));
    rhs_.reset(l_form_->ParallelAssemble());
  } else {
    rhs_  = temperature_.createOnSpace<mfem::HypreParVector>();
    *rhs_ = 0.0;
  }

  // Build the dof array lookup tables
  temperature_.space().BuildDofToArrays();

  // Project the essential boundary coefficients
  for (auto& bc : bcs_.essentials()) {
    bc.projectBdr(temperature_, time_);
  }

  // Assemble the stiffness matrix
  K_.reset(K_form_->ParallelAssemble());

  // Initialize the eliminated BC RHS vector
  bc_rhs_  = temperature_.createOnSpace<mfem::HypreParVector>();
  *bc_rhs_ = 0.0;

  // Initialize the true vector
  temperature_.initializeTrueVec();

  if (is_quasistatic_) {
    residual_ = mfem_ext::StdFunctionOperator(
        temperature_.space().TrueVSize(),

        [this](const mfem::Vector& u, mfem::Vector& r) {
          r = (*K_) * u;
          r.SetSubVector(bcs_.allEssentialDofs(), 0.0);
        },

        [this](const mfem::Vector & /*du_dt*/) -> mfem::Operator& {
          if (J_ == nullptr) {
            J_.reset(K_form_->ParallelAssemble());
            bcs_.eliminateAllEssentialDofsFromMatrix(*J_);
          }
          return *J_;
        });

  } else {
    // If dynamic, assemble the mass matrix
    M_form_ = temperature_.createOnSpace<mfem::ParBilinearForm>();

    // Define the mass matrix coefficient as a product of the density and specific heat capacity
    mass_coef_ = std::make_unique<mfem::ProductCoefficient>(*rho_, *cp_);

    M_form_->AddDomainIntegrator(new mfem::MassIntegrator(*mass_coef_));
    M_form_->Assemble(0);  // keep sparsity pattern of M and K the same
    M_form_->Finalize();

    M_.reset(M_form_->ParallelAssemble());

    residual_ = mfem_ext::StdFunctionOperator(
        temperature_.space().TrueVSize(),
        [this](const mfem::Vector& du_dt, mfem::Vector& r) {
          r = (*M_) * du_dt + (*K_) * (u_ + dt_ * du_dt);
          r.SetSubVector(bcs_.allEssentialDofs(), 0.0);
        },

        [this](const mfem::Vector & /*du_dt*/) -> mfem::Operator& {
          if (dt_ != previous_dt_) {
            J_.reset(mfem::Add(1.0, *M_, dt_, *K_));
            bcs_.eliminateAllEssentialDofsFromMatrix(*J_);
          }
          return *J_;
        });
  }
}

void ThermalConduction::advanceTimestep(double& dt)
{
  temperature_.initializeTrueVec();

  if (is_quasistatic_) {
    nonlin_solver_.Mult(zero_, temperature_.trueVec());
  } else {
    SLIC_ASSERT_MSG(gf_initialized_[0], "Thermal state not initialized!");

    // Step the time integrator
    ode_.Step(temperature_.trueVec(), time_, dt);
  }

  temperature_.distributeSharedDofs();
  cycle_ += 1;
}

}  // namespace serac

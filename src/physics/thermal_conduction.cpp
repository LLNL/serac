// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "physics/thermal_conduction.hpp"

#include "infrastructure/logger.hpp"

namespace serac {

constexpr int NUM_FIELDS = 1;

ThermalConduction::ThermalConduction(int order, std::shared_ptr<mfem::ParMesh> mesh, const SolverParameters& params)
    : BasePhysics(mesh, NUM_FIELDS, order),
      temperature_(*mesh,
                   FiniteElementState::Options{
                       .order = order, .space_dim = 1, .ordering = mfem::Ordering::byNODES, .name = "temperature"}),
      residual_(temperature_.space().TrueVSize())
{
  state_.push_back(temperature_);

  nonlin_solver_ = EquationSolver(mesh->GetComm(), params.T_lin_params, params.T_nonlin_params);
  nonlin_solver_.SetOperator(residual_);

  // Check for dynamic mode
  if (params.dyn_params) {
    setTimestepper(params.dyn_params->timestepper, params.dyn_params->enforcement_method);
  } else {
    setTimestepper(TimestepMethod::QuasiStatic);
  }

  dt_          = 0.0;
  previous_dt_ = -1.0;

  int true_size = temperature_.space().TrueVSize();
  u_.SetSize(true_size);
  previous_.SetSize(true_size);
  previous_ = 0.0;

  zero_.SetSize(true_size);
  zero_ = 0.0;

  U_minus_.SetSize(true_size);
  U_.SetSize(true_size);
  U_plus_.SetSize(true_size);
  dU_dt_.SetSize(true_size);

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

void ThermalConduction::setConductivity(CoefficientWrapper && kappa)
{
  // Set the conduction coefficient
  kappa_ = std::move(kappa);
}

void ThermalConduction::setSource(CoefficientWrapper&& source)
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
  K_form_->AddDomainIntegrator(new mfem::DiffusionIntegrator(kappa_));
  K_form_->Assemble(0);  // keep sparsity pattern of M and K the same
  K_form_->Finalize();

  // Add the body source to the RS if specified
  l_form_ = temperature_.createOnSpace<mfem::ParLinearForm>();
  if (source_) {
    l_form_->AddDomainIntegrator(new mfem::DomainLFIntegrator(source_));
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

  if (timestepper_ == serac::TimestepMethod::QuasiStatic) {
    residual_ = StdFunctionOperator(
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
    //mass_coef_ = std::make_unique<mfem::ProductCoefficient>(*rho_, *cp_);

    M_form_->AddDomainIntegrator(new mfem::MassIntegrator(mfem::ProductCoefficient(rho_, cp_)));
    M_form_->Assemble(0);  // keep sparsity pattern of M and K the same
    M_form_->Finalize();

    M_.reset(M_form_->ParallelAssemble());

    residual_ = StdFunctionOperator(
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

    ode_ = FirstOrderODE(temperature_.trueVec().Size(), [this](const double t, const double dt, const mfem::Vector& u,
                                                               mfem::Vector& du_dt) {
      // this is intended to be temporary
      // Ideally, epsilon should be "small" relative to the characteristic
      // time of the ODE, but we can't ensure that at present (we don't have
      // a critical timestep estimate)
      constexpr double epsilon = 0.0001;

      // assign these values to variables with greater scope,
      // so that the residual operator can see them
      dt_ = dt;
      u_  = u;

      // TODO: take care of this last part of the ODE definition
      //       automatically by wrapping mfem's ODE solvers
      //
      // evaluate the constraint functions at a 3-point
      // stencil of times centered on the time of interest
      // in order to compute finite-difference approximations
      // to the time derivatives that appear in the residual
      U_minus_ = 0.0;
      U_       = 0.0;
      U_plus_  = 0.0;
      for (const auto& bc : bcs_.essentials()) {
        bc.projectBdrToDofs(U_minus_, t - epsilon);
        bc.projectBdrToDofs(U_, t);
        bc.projectBdrToDofs(U_plus_, t + epsilon);
      }

      bool implicit = (dt != 0.0);
      if (implicit) {
        if (enforcement_method_ == DirichletEnforcementMethod::DirectControl) {
          dU_dt_ = (U_ - u) / dt;
          U_     = u;
        }

        if (enforcement_method_ == DirichletEnforcementMethod::RateControl) {
          dU_dt_ = (U_plus_ - U_minus_) / (2.0 * epsilon);
          U_     = u;
        }

        if (enforcement_method_ == DirichletEnforcementMethod::FullControl) {
          dU_dt_ = (U_plus_ - U_minus_) / (2.0 * epsilon);
          U_     = U_ - dt * dU_dt_;
        }
      } else {
        dU_dt_ = (U_plus_ - U_minus_) / (2.0 * epsilon);
      }

      auto constrained_dofs = bcs_.allEssentialDofs();
      u_.SetSubVector(constrained_dofs, 0.0);
      U_.SetSubVectorComplement(constrained_dofs, 0.0);
      u_ += U_;

      du_dt = previous_;
      du_dt.SetSubVector(constrained_dofs, 0.0);
      dU_dt_.SetSubVectorComplement(constrained_dofs, 0.0);
      du_dt += dU_dt_;

      nonlin_solver_.Mult(zero_, du_dt);
      SLIC_WARNING_IF(!nonlin_solver_.nonlinearSolver().GetConverged(), "Newton Solver did not converge.");

      previous_    = du_dt;
      previous_dt_ = dt;
    });

    ode_solver_->Init(ode_);
  }
}

void ThermalConduction::advanceTimestep(double& dt)
{
  temperature_.initializeTrueVec();

  if (timestepper_ == serac::TimestepMethod::QuasiStatic) {
    nonlin_solver_.Mult(zero_, temperature_.trueVec());
  } else {
    SLIC_ASSERT_MSG(gf_initialized_[0], "Thermal state not initialized!");

    // Step the time integrator
    ode_solver_->Step(temperature_.trueVec(), time_, dt);
  }

  temperature_.distributeSharedDofs();
  cycle_ += 1;
}

}  // namespace serac

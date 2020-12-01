// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "physics/nonlinear_solid.hpp"

#include "infrastructure/logger.hpp"
#include "integrators/hyperelastic_traction_integrator.hpp"
#include "integrators/inc_hyperelastic_integrator.hpp"
#include "numerics/mesh_utils.hpp"

namespace serac {

constexpr int NUM_FIELDS = 2;

NonlinearSolid::NonlinearSolid(int order, std::shared_ptr<mfem::ParMesh> mesh, const SolverParameters& params)
    : BasePhysics(mesh, NUM_FIELDS, order),
      velocity_(*mesh, FiniteElementState::Options{.order = order, .name = "velocity"}),
      displacement_(*mesh, FiniteElementState::Options{.order = order, .name = "displacement"}),
      residual_(displacement_.space().TrueVSize())
{
  state_.push_back(velocity_);
  state_.push_back(displacement_);

  // Initialize the mesh node pointers
  reference_nodes_ = displacement_.createOnSpace<mfem::ParGridFunction>();
  mesh->GetNodes(*reference_nodes_);
  mesh->NewNodes(*reference_nodes_);

  reference_nodes_->GetTrueDofs(x_);
  deformed_nodes_ = std::make_unique<mfem::ParGridFunction>(*reference_nodes_);

  displacement_.trueVec() = 0.0;
  velocity_.trueVec()     = 0.0;

  const auto& lin_params = params.H_lin_params;
  // If the user wants the AMG preconditioner with a linear solver, set the pfes
  // to be the displacement
  const auto& augmented_params = augmentAMGForElasticity(lin_params, displacement_.space());

  nonlin_solver_ = EquationSolver(mesh->GetComm(), augmented_params, params.H_nonlin_params);

  // Check for dynamic mode
  if (params.dyn_params) {
    setTimestepper(params.dyn_params->timestepper, params.dyn_params->enforcement_method);
  } else {
    setTimestepper(TimestepMethod::QuasiStatic);
  }

  int true_size = velocity_.space().TrueVSize();

  u_.SetSize(true_size);
  du_dt_.SetSize(true_size);
  previous_.SetSize(true_size);
  previous_ = 0.0;

  zero_.SetSize(true_size);
  zero_ = 0.0;

  U_minus_.SetSize(true_size);
  U_.SetSize(true_size);
  U_plus_.SetSize(true_size);
  dU_dt_.SetSize(true_size);
  d2U_dt2_.SetSize(true_size);
}

NonlinearSolid::NonlinearSolid(std::shared_ptr<mfem::ParMesh> mesh, const NonlinearSolid::InputInfo& info)
    : NonlinearSolid(info.order, mesh, info.solver_params)
{
  // This is the only other info stored in the input file that we can use
  // in the initialization stage
  setHyperelasticMaterialParameters(info.mu, info.K);
}

void NonlinearSolid::setDisplacementBCs(const std::set<int>&                     disp_bdr,
                                        std::shared_ptr<mfem::VectorCoefficient> disp_bdr_coef)
{
  bcs_.addEssential(disp_bdr, disp_bdr_coef, displacement_, -1);
}

void NonlinearSolid::setDisplacementBCs(const std::set<int>& disp_bdr, std::shared_ptr<mfem::Coefficient> disp_bdr_coef,
                                        int component)
{
  bcs_.addEssential(disp_bdr, disp_bdr_coef, displacement_, component);
}

void NonlinearSolid::setTractionBCs(const std::set<int>&                     trac_bdr,
                                    std::shared_ptr<mfem::VectorCoefficient> trac_bdr_coef, int component)
{
  bcs_.addNatural(trac_bdr, trac_bdr_coef, component);
}

void NonlinearSolid::setHyperelasticMaterialParameters(const double mu, const double K)
{
  model_ = std::make_unique<mfem::NeoHookeanModel>(mu, K);
}

void NonlinearSolid::setViscosity(coefficient visc_coef) { viscosity_ = std::move(visc_coef); }

void NonlinearSolid::setDisplacement(mfem::VectorCoefficient& disp_state)
{
  disp_state.SetTime(time_);
  displacement_.project(disp_state);
  gf_initialized_[1] = true;
}

void NonlinearSolid::setVelocity(mfem::VectorCoefficient& velo_state)
{
  velo_state.SetTime(time_);
  velocity_.project(velo_state);
  gf_initialized_[0] = true;
}

void NonlinearSolid::completeSetup()
{
  // Define the nonlinear form
  H_ = displacement_.createOnSpace<mfem::ParNonlinearForm>();

  // Add the hyperelastic integrator
  if (timestepper_ == serac::TimestepMethod::QuasiStatic) {
    H_->AddDomainIntegrator(new IncrementalHyperelasticIntegrator(model_.get()));
  } else {
    H_->AddDomainIntegrator(new mfem::HyperelasticNLFIntegrator(model_.get()));
  }

  // Add the traction integrator
  for (auto& nat_bc_data : bcs_.naturals()) {
    H_->AddBdrFaceIntegrator(new HyperelasticTractionIntegrator(nat_bc_data.vectorCoefficient()),
                             nat_bc_data.markers());
  }

  // Build the dof array lookup tables
  displacement_.space().BuildDofToArrays();

  // Project the essential boundary coefficients
  for (auto& bc : bcs_.essentials()) {
    // Project the coefficient
    bc.project(displacement_);
  }

  // If dynamic, create the mass and viscosity forms
  if (timestepper_ != serac::TimestepMethod::QuasiStatic) {
    const double              ref_density = 1.0;  // density in the reference configuration
    mfem::ConstantCoefficient rho0(ref_density);

    M_ = displacement_.createOnSpace<mfem::ParBilinearForm>();
    M_->AddDomainIntegrator(new mfem::VectorMassIntegrator(rho0));
    M_->Assemble(0);
    M_->Finalize(0);

    M_mat_.reset(M_->ParallelAssemble());

    C_ = displacement_.createOnSpace<mfem::ParBilinearForm>();
    C_->AddDomainIntegrator(new mfem::VectorDiffusionIntegrator(viscosity_));
    C_->Assemble(0);
    C_->Finalize(0);

    C_mat_.reset(C_->ParallelAssemble());
  }

  // We are assuming that the ODE is prescribing the
  // acceleration value for the constrained dofs, so
  // the residuals for those dofs can be taken to be zero.
  //
  // Setting iterative_mode to true ensures that these
  // prescribed acceleration values are not modified by
  // the nonlinear solve.
  nonlin_solver_.nonlinearSolver().iterative_mode = true;
  nonlin_solver_.SetOperator(residual_);

  if (timestepper_ == serac::TimestepMethod::QuasiStatic) {
    // the quasistatic case is entirely described by the residual,
    // there is no ordinary differential equation
    residual_ = StdFunctionOperator(
        displacement_.space().TrueVSize(),

        // residual function
        [this](const mfem::Vector& u, mfem::Vector& r) {
          H_->Mult(u, r);  // r := H(u)
          r.SetSubVector(bcs_.allEssentialDofs(), 0.0);
        },

        // gradient of residual function
        [this](const mfem::Vector& u) -> mfem::Operator& {
          auto& J = dynamic_cast<mfem::HypreParMatrix&>(H_->GetGradient(u));
          bcs_.eliminateAllEssentialDofsFromMatrix(J);
          return J;
        });

  } else {
    // the dynamic case is described by a residual function and a second order
    // ordinary differential equation. Here, we define the residual function in
    // terms of an acceleration.
    residual_ = StdFunctionOperator(
        displacement_.space().TrueVSize(),

        // residual function
        [this](const mfem::Vector& d2u_dt2, mfem::Vector& r) {
          r = (*M_mat_) * d2u_dt2 + (*C_mat_) * (du_dt_ + c1_ * d2u_dt2) + (*H_) * (x_ + u_ + c0_ * d2u_dt2);
          r.SetSubVector(bcs_.allEssentialDofs(), 0.0);
        },

        // gradient of residual function
        [this](const mfem::Vector& d2u_dt2) -> mfem::Operator& {
          // J = M + c1 * C + c0 * H(u_predicted)
          auto localJ = std::unique_ptr<mfem::SparseMatrix>(Add(1.0, M_->SpMat(), c1_, C_->SpMat()));
          localJ->Add(c0_, H_->GetLocalGradient(x_ + u_ + c0_ * d2u_dt2));
          J_mat_.reset(M_->ParallelAssemble(localJ.get()));
          bcs_.eliminateAllEssentialDofsFromMatrix(*J_mat_);
          return *J_mat_;
        });

    ode2_ = SecondOrderODE(
        u_.Size(), [this](const double t, const double fac0, const double fac1, const mfem::Vector& displacement,
                          const mfem::Vector& velocity, mfem::Vector& acceleration) {
          // this is intended to be temporary
          // Ideally, epsilon should be "small" relative to the characteristic time
          // of the ODE, but we can't ensure that at present (we don't have a
          // critical timestep estimate)
          constexpr double epsilon = 0.0001;

          // assign these values to variables with greater scope,
          // so that the residual operator can see them
          c0_    = fac0;
          c1_    = fac1;
          u_     = displacement;
          du_dt_ = velocity;

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

          bool implicit = (c0_ != 0.0 || c1_ != 0.0);
          if (implicit) {
            if (enforcement_method_ == DirichletEnforcementMethod::DirectControl) {
              d2U_dt2_ = (U_ - u_) / c0_;
              dU_dt_   = du_dt_;
              U_       = u_;
            }

            if (enforcement_method_ == DirichletEnforcementMethod::RateControl) {
              d2U_dt2_ = (dU_dt_ - du_dt_) / c1_;
              dU_dt_   = du_dt_;
              U_       = u_;
            }

            if (enforcement_method_ == DirichletEnforcementMethod::FullControl) {
              d2U_dt2_ = (U_minus_ - 2.0 * U_ + U_plus_) / (epsilon * epsilon);
              dU_dt_   = (U_plus_ - U_minus_) / (2.0 * epsilon) - c1_ * d2U_dt2_;
              U_       = U_ - c0_ * d2U_dt2_;
            }
          } else {
            d2U_dt2_ = (U_minus_ - 2.0 * U_ + U_plus_) / (epsilon * epsilon);
            dU_dt_   = (U_plus_ - U_minus_) / (2.0 * epsilon);
          }

          auto constrained_dofs = bcs_.allEssentialDofs();
          u_.SetSubVector(constrained_dofs, 0.0);
          U_.SetSubVectorComplement(constrained_dofs, 0.0);
          u_ += U_;

          du_dt_.SetSubVector(constrained_dofs, 0.0);
          dU_dt_.SetSubVectorComplement(constrained_dofs, 0.0);
          du_dt_ += dU_dt_;

          // use the previous solution as our starting guess
          acceleration = previous_;
          acceleration.SetSubVector(constrained_dofs, 0.0);
          d2U_dt2_.SetSubVectorComplement(constrained_dofs, 0.0);
          acceleration += d2U_dt2_;

          nonlin_solver_.Mult(zero_, acceleration);
          SLIC_WARNING_IF(!nonlin_solver_.nonlinearSolver().GetConverged(), "Newton Solver did not converge.");

          previous_ = acceleration;
        });

    second_order_ode_solver_->Init(ode2_);
  }
}

// Advance the timestep
void NonlinearSolid::advanceTimestep(double& dt)
{
  // Initialize the true vector
  velocity_.initializeTrueVec();
  displacement_.initializeTrueVec();

  // Set the mesh nodes to the reference configuration
  mesh_->NewNodes(*reference_nodes_);

  if (timestepper_ == serac::TimestepMethod::QuasiStatic) {
    nonlin_solver_.Mult(zero_, displacement_.trueVec());
  } else {
    second_order_ode_solver_->Step(displacement_.trueVec(), velocity_.trueVec(), time_, dt);
  }

  // Distribute the shared DOFs
  velocity_.distributeSharedDofs();
  displacement_.distributeSharedDofs();

  // Update the mesh with the new deformed nodes
  deformed_nodes_->Set(1.0, displacement_.gridFunc());
  deformed_nodes_->Add(1.0, *reference_nodes_);

  mesh_->NewNodes(*deformed_nodes_);

  cycle_ += 1;
}

NonlinearSolid::~NonlinearSolid() {}

void NonlinearSolid::InputInfo::defineInputFileSchema(axom::inlet::Table& table)
{
  // Polynomial interpolation order
  table.addInt("order", "Order degree of the finite elements.").defaultValue(1);

  // neo-Hookean material parameters
  table.addDouble("mu", "Shear modulus in the Neo-Hookean hyperelastic model.").defaultValue(0.25);
  table.addDouble("K", "Bulk modulus in the Neo-Hookean hyperelastic model.").defaultValue(5.0);

  auto& traction_table = table.addTable("traction", "Cantilever tip traction vector");

  // loading parameters
  input::defineVectorInputFileSchema(traction_table);

  traction_table.getField("x").defaultValue(0.0);
  traction_table.getField("y").defaultValue(1.0e-3);
  traction_table.getField("z").defaultValue(0.0);

  auto& solver_table = table.addTable("solver", "Linear and Nonlinear Solver Parameters.");
  serac::EquationSolver::defineInputFileSchema(solver_table);

  auto& bc_table = table.addGenericArray("boundary_conds", "Boundary condition information");
  serac::input::BoundaryConditionInputInfo::defineInputFileSchema(bc_table);
}

}  // namespace serac

using serac::NonlinearSolid;

NonlinearSolid::InputInfo FromInlet<NonlinearSolid::InputInfo>::operator()(const axom::inlet::Table& base)
{
  NonlinearSolid::InputInfo result;

  result.order = base["order"];

  // Solver parameters
  auto solver                          = base["solver"];
  result.solver_params.H_lin_params    = solver["linear"].get<serac::IterativeSolverParameters>();
  result.solver_params.H_nonlin_params = solver["nonlinear"].get<serac::NonlinearSolverParameters>();

  // TODO: "optional" concept within Inlet to support dynamic parameters

  // Set the material parameters
  // neo-Hookean material parameters
  result.mu = base["mu"];
  result.K  = base["K"];

  auto bdr_map = base["boundary_conds"].get<std::unordered_map<int, serac::input::BoundaryConditionInputInfo>>();
  for (const auto& [idx, val] : bdr_map) {
    result.boundary_conditions.push_back(val);
  }

  return result;
}

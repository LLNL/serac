// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "physics/nonlinear_solid_solver.hpp"

#include "infrastructure/logger.hpp"
#include "integrators/hyperelastic_traction_integrator.hpp"
#include "integrators/inc_hyperelastic_integrator.hpp"
#include "numerics/expr_template_ops.hpp"

namespace serac {

constexpr int           NUM_FIELDS = 2;
static constexpr double epsilon    = 0.001;

NonlinearSolidSolver::NonlinearSolidSolver(int order, std::shared_ptr<mfem::ParMesh> mesh)
    : BaseSolver(mesh, NUM_FIELDS, order),
      velocity_(std::make_shared<FiniteElementState>(*mesh, FEStateOptions{.order = order, .name = "velocity"})),
      displacement_(
          std::make_shared<FiniteElementState>(*mesh, FEStateOptions{.order = order, .name = "displacement"})),
      op(displacement_->space().TrueVSize())
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

  u     = mfem::Vector(true_size);
  du_dt = mfem::Vector(true_size);
  zero  = mfem::Vector(true_size);
  zero  = 0.0;

  U_minus = mfem::Vector(true_size);
  U       = mfem::Vector(true_size);
  U_plus  = mfem::Vector(true_size);
  dU_dt   = mfem::Vector(true_size);
  d2U_dt2 = mfem::Vector(true_size);

  x = *reference_nodes_;
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
  H           = displacement_->createOnSpace<mfem::ParNonlinearForm>();

  // Add the hyperelastic integrator
  if (timestepper_ == serac::TimestepMethod::QuasiStatic) {
    H_form->AddDomainIntegrator(new IncrementalHyperelasticIntegrator(model_.get()));
    H->AddDomainIntegrator(new IncrementalHyperelasticIntegrator(model_.get()));
  } else {
    H_form->AddDomainIntegrator(new mfem::HyperelasticNLFIntegrator(model_.get()));
    H->AddDomainIntegrator(new mfem::HyperelasticNLFIntegrator(model_.get()));
  }

  // Add the traction integrator
  for (auto& nat_bc_data : bcs_.naturals()) {
    H_form->AddBdrFaceIntegrator(new HyperelasticTractionIntegrator(nat_bc_data.vectorCoefficient()),
                                 nat_bc_data.markers());
    H->AddBdrFaceIntegrator(new HyperelasticTractionIntegrator(nat_bc_data.vectorCoefficient()), nat_bc_data.markers());
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

    M = displacement_->createOnSpace<mfem::ParBilinearForm>();
    M->AddDomainIntegrator(new mfem::VectorMassIntegrator(rho0));
    M->Assemble(0);
    M->Finalize(0);

    S_form = displacement_->createOnSpace<mfem::ParBilinearForm>();
    S_form->AddDomainIntegrator(new mfem::VectorDiffusionIntegrator(*viscosity_));
    S_form->Assemble(0);
    S_form->Finalize(0);

    S = displacement_->createOnSpace<mfem::ParBilinearForm>();
    S->AddDomainIntegrator(new mfem::VectorDiffusionIntegrator(*viscosity_));
    S->Assemble(0);
    S->Finalize(0);
  }

  solver_ = EquationSolver(displacement_->comm(), lin_params_, nonlin_params_);
  // Set up the jacbian solver based on the linear solver options
  if (lin_params_.prec == serac::Preconditioner::BoomerAMG) {
    SLIC_WARNING_IF(displacement_->space().GetOrdering() == mfem::Ordering::byNODES,
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
    solver_.nonlinearSolver().iterative_mode = true;
    nonlinear_oper_ = std::make_unique<NonlinearSolidQuasiStaticOperator>(std::move(H_form), bcs_);
    solver_.SetOperator(*nonlinear_oper_);
  } else {
    solver_.nonlinearSolver().iterative_mode = false;
    timedep_oper_ = std::make_unique<NonlinearSolidDynamicOperator>(std::move(H_form), std::move(S_form),
                                                                    std::move(M_form), bcs_, solver_, lin_params_);
    ode_solver_->Init(*timedep_oper_);

    root_finder          = EquationSolver(displacement_->comm(), lin_params_, nonlin_params_);
    auto J_hypreSmoother = std::make_unique<mfem::HypreSmoother>();
    J_hypreSmoother->SetType(mfem::HypreSmoother::l1Jacobi);
    J_hypreSmoother->SetPositiveDiagonal(true);
    root_finder.SetPreconditioner(std::move(J_hypreSmoother));

    // We are assuming that the ODE is prescribing the
    // acceleration value for the constrained dofs, so
    // the residuals for those dofs can be taken to be zero.
    // 
    // Setting iterative_mode to true ensures that these
    // prescribed acceleration values are not modified by
    // the nonlinear solve.
    root_finder.nonlinearSolver().iterative_mode = true;

    root_finder.SetOperator(op);

    op.residual = [=](const mfem::Vector& d2u_dt2, mfem::Vector& res) mutable {
      res = (*M) * d2u_dt2 + (*S) * (du_dt + c1 * d2u_dt2) + (*H) * (x + u + c0 * d2u_dt2);
      res.SetSubVector(bcs_.allEssentialDofs(), 0.0);
    };

    op.jacobian = [=](const mfem::Vector& d2u_dt2) mutable -> mfem::Operator& {
      // J = M + dt1 * S + 0.5 * dt0 * dt0 * H(u_predicted)
      auto localJ = std::unique_ptr<mfem::SparseMatrix>(Add(1.0, M->SpMat(), c1, S->SpMat()));
      localJ->Add(c0, H->GetLocalGradient(x + u + c0 * d2u_dt2));
      J.reset(M->ParallelAssemble(localJ.get()));
      bcs_.eliminateAllEssentialDofsFromMatrix(*J);
      return *J;
    };

    ode2 = SecondOrderODE(
        u.Size(), [=](const double t, const double fac0, const double fac1, const mfem::Vector& displacement,
                      const mfem::Vector& velocity, mfem::Vector& acceleration) {
          // pass these values through to the physics module
          // so that the residual operator can see them
          c0 = fac0;
          c1 = fac1;
          u = displacement;
          du_dt = velocity;

          // evaluate the constraint functions at a 3-point
          // stencil of times centered on the time of interest
          // in order to compute finite-difference approximations
          // to the time derivatives that appear in the residual
          U_minus = 0.0;
          U       = 0.0;
          U_plus  = 0.0;
          for (const auto& bc : bcs_.essentials()) {
            bc.projectBdrToDofs(U_minus, t - epsilon);
            bc.projectBdrToDofs(U, t);
            bc.projectBdrToDofs(U_plus, t + epsilon);
          }

          bool implicit = (c0 != 0.0 || c1 != 0.0);
          if (implicit) {
            if (enforcement_method_ == DirichletEnforcementMethod::DirectControl) {
              d2U_dt2 = (U - u) / c0;
              dU_dt   = du_dt;
              U       = u;
            }

            if (enforcement_method_ == DirichletEnforcementMethod::RateControl) {
              d2U_dt2 = (dU_dt - du_dt) / c1;
              dU_dt   = du_dt;
              U       = u;
            }

            if (enforcement_method_ == DirichletEnforcementMethod::FullControl) {
              d2U_dt2 = (U_minus - 2.0 * U + U_plus) / (epsilon * epsilon);
              dU_dt   = (U_minus - U_plus) / (2.0 * epsilon) - c1 * d2U_dt2;
              U       = U - c0 * d2U_dt2;
            }
          } else {
            d2U_dt2 = (U_minus - 2.0 * U + U_plus) / (epsilon * epsilon);
            dU_dt   = (U_minus - U_plus) / (2.0 * epsilon);
          }

          auto constrained_dofs = bcs_.allEssentialDofs();
          u.SetSubVector(constrained_dofs, 0.0);
          U.SetSubVectorComplement(constrained_dofs, 0.0);
          u += U;

          du_dt.SetSubVector(constrained_dofs, 0.0);
          dU_dt.SetSubVectorComplement(constrained_dofs, 0.0);
          du_dt += dU_dt;

          acceleration.SetSubVector(constrained_dofs, 0.0);
          d2U_dt2.SetSubVectorComplement(constrained_dofs, 0.0);
          acceleration += d2U_dt2;

          root_finder.Mult(zero, acceleration);
          SLIC_WARNING_IF(!root_finder.nonlinearSolver().GetConverged(), "Newton Solver did not converge.");
        });

    second_order_ode_solver_->Init(ode2);
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
    second_order_ode_solver_->Step(displacement_->trueVec(), velocity_->trueVec(), time_, dt);
  }

  // Distribute the shared DOFs
  velocity_->distributeSharedDofs();
  displacement_->distributeSharedDofs();

  // Update the mesh with the new deformed nodes
  deformed_nodes_->Set(1.0, displacement_->gridFunc());
  deformed_nodes_->Add(1.0, *reference_nodes_);
  mesh_->NewNodes(*deformed_nodes_);
  
  // x = *deformed_nodes_;

  cycle_ += 1;
}

NonlinearSolidSolver::~NonlinearSolidSolver() {}

}  // namespace serac
// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/nonlinear_solid.hpp"

#include "serac/infrastructure/logger.hpp"
#include "serac/integrators/hyperelastic_traction_integrator.hpp"
#include "serac/integrators/inc_hyperelastic_integrator.hpp"
#include "serac/integrators/wrapper_integrator.hpp"
#include "serac/numerics/expr_template_ops.hpp"
#include "serac/numerics/mesh_utils.hpp"

namespace serac {

constexpr int NUM_FIELDS = 2;

NonlinearSolid::NonlinearSolid(int order, std::shared_ptr<mfem::ParMesh> mesh, const SolverOptions& options)
    : BasePhysics(mesh, NUM_FIELDS, order),
      velocity_(*mesh, FiniteElementState::Options{.order = order, .name = "velocity"}),
      displacement_(*mesh, FiniteElementState::Options{.order = order, .name = "displacement"}),
      ode2_(displacement_.space().TrueVSize(), {.c0 = c0_, .c1 = c1_, .u = u_, .du_dt = du_dt_, .d2u_dt2 = previous_},
            nonlin_solver_, bcs_)
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

  const auto& lin_options = options.H_lin_options;
  // If the user wants the AMG preconditioner with a linear solver, set the pfes
  // to be the displacement
  const auto& augmented_options = mfem_ext::AugmentAMGForElasticity(lin_options, displacement_.space());

  nonlin_solver_ = mfem_ext::EquationSolver(mesh->GetComm(), augmented_options, options.H_nonlin_options);

  // Check for dynamic mode
  if (options.dyn_options) {
    ode2_.SetTimestepper(options.dyn_options->timestepper);
    ode2_.SetEnforcementMethod(options.dyn_options->enforcement_method);
    is_quasistatic_ = false;
  } else {
    is_quasistatic_ = true;
  }

  int true_size = velocity_.space().TrueVSize();

  u_.SetSize(true_size);
  du_dt_.SetSize(true_size);
  previous_.SetSize(true_size);
  previous_ = 0.0;

  zero_.SetSize(true_size);
  zero_ = 0.0;
}

NonlinearSolid::NonlinearSolid(std::shared_ptr<mfem::ParMesh> mesh, const NonlinearSolid::InputOptions& options)
    : NonlinearSolid(options.order, mesh, options.solver_options)
{
  // This is the only other options stored in the input file that we can use
  // in the initialization stage
  setHyperelasticMaterialParameters(options.mu, options.K);

  auto dim = mesh->Dimension();
  if (options.initial_displacement) {
    auto deform = options.initial_displacement->constructVector(dim);
    setDisplacement(deform);
  }

  if (options.initial_velocity) {
    auto velo = options.initial_velocity->constructVector(dim);
    setVelocity(velo);
  }
  for (const auto& [name, bc] : options.boundary_conditions) {
    // FIXME: Better naming for boundary conditions?
    if (name.find("displacement") != std::string::npos) {
      if (bc.coef_opts.isVector()) {
        auto disp_coef = std::make_shared<mfem::VectorFunctionCoefficient>(bc.coef_opts.constructVector(dim));
        setDisplacementBCs(bc.attrs, disp_coef);
      } else {
        auto disp_coef = std::make_shared<mfem::FunctionCoefficient>(bc.coef_opts.constructScalar());
        setDisplacementBCs(bc.attrs, disp_coef, bc.coef_opts.component);
      }
    } else if (name.find("traction") != std::string::npos) {
      auto trac_coef = std::make_shared<mfem::VectorFunctionCoefficient>(bc.coef_opts.constructVector(dim));
      setTractionBCs(bc.attrs, trac_coef);
    } else {
      SLIC_WARNING("Ignoring boundary condition with unknown name: " << name);
    }
  }
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

void NonlinearSolid::addBodyForce(std::shared_ptr<mfem::VectorCoefficient> ext_force_coef)
{
  ext_force_coefs_.push_back(ext_force_coef);
}

void NonlinearSolid::setHyperelasticMaterialParameters(const double mu, const double K)
{
  model_ = std::make_unique<mfem::NeoHookeanModel>(mu, K);
}

void NonlinearSolid::setViscosity(std::unique_ptr<mfem::Coefficient>&& visc_coef) { viscosity_ = std::move(visc_coef); }

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
  if (is_quasistatic_) {
    H_->AddDomainIntegrator(new mfem_ext::IncrementalHyperelasticIntegrator(model_.get()));
  } else {
    H_->AddDomainIntegrator(new mfem::HyperelasticNLFIntegrator(model_.get()));
  }

  // Add the traction integrator
  for (auto& nat_bc_data : bcs_.naturals()) {
    H_->AddBdrFaceIntegrator(new mfem_ext::HyperelasticTractionIntegrator(nat_bc_data.vectorCoefficient()),
                             nat_bc_data.markers());
  }

  // Add external forces
  for (auto& force : ext_force_coefs_) {
    H_->AddDomainIntegrator(new serac::mfem_ext::LinearToNonlinearFormIntegrator(
        std::make_shared<mfem::VectorDomainLFIntegrator>(*force),
        std::make_shared<mfem::ParFiniteElementSpace>(*H_->ParFESpace())));
  }

  // Build the dof array lookup tables
  displacement_.space().BuildDofToArrays();

  // Project the essential boundary coefficients
  for (auto& bc : bcs_.essentials()) {
    // Project the coefficient
    bc.project(displacement_);
  }

  // If dynamic, create the mass and viscosity forms
  if (!is_quasistatic_) {
    const double              ref_density = 1.0;  // density in the reference configuration
    mfem::ConstantCoefficient rho0(ref_density);

    M_ = displacement_.createOnSpace<mfem::ParBilinearForm>();
    M_->AddDomainIntegrator(new mfem::VectorMassIntegrator(rho0));
    M_->Assemble(0);
    M_->Finalize(0);

    M_mat_.reset(M_->ParallelAssemble());

    C_ = displacement_.createOnSpace<mfem::ParBilinearForm>();
    C_->AddDomainIntegrator(new mfem::VectorDiffusionIntegrator(*viscosity_));
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
  nonlin_solver_.NonlinearSolver().iterative_mode = true;

  if (is_quasistatic_) {
    residual_ = buildQuasistaticOperator();

  } else {
    // the dynamic case is described by a residual function and a second order
    // ordinary differential equation. Here, we define the residual function in
    // terms of an acceleration.
    residual_ = std::make_unique<mfem_ext::StdFunctionOperator>(
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
  }

  nonlin_solver_.SetOperator(*residual_);
}

// Solve the Quasi-static Newton system
void NonlinearSolid::quasiStaticSolve() { nonlin_solver_.Mult(zero_, displacement_.trueVec()); }

std::unique_ptr<mfem::Operator> NonlinearSolid::buildQuasistaticOperator()
{
  // the quasistatic case is entirely described by the residual,
  // there is no ordinary differential equation
  auto residual = std::make_unique<mfem_ext::StdFunctionOperator>(
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
  return residual;
}

// Advance the timestep
void NonlinearSolid::advanceTimestep(double& dt)
{
  // Initialize the true vector
  velocity_.initializeTrueVec();
  displacement_.initializeTrueVec();

  // Set the mesh nodes to the reference configuration
  mesh_->NewNodes(*reference_nodes_);

  if (is_quasistatic_) {
    quasiStaticSolve();
  } else {
    ode2_.Step(displacement_.trueVec(), velocity_.trueVec(), time_, dt);
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

void NonlinearSolid::InputOptions::defineInputFileSchema(axom::inlet::Table& table)
{
  // Polynomial interpolation order - currently up to 8th order is allowed
  table.addInt("order", "Order degree of the finite elements.").defaultValue(1).range(1, 8);

  // neo-Hookean material parameters
  table.addDouble("mu", "Shear modulus in the Neo-Hookean hyperelastic model.").defaultValue(0.25);
  table.addDouble("K", "Bulk modulus in the Neo-Hookean hyperelastic model.").defaultValue(5.0);

  auto& stiffness_solver_table =
      table.addTable("stiffness_solver", "Linear and Nonlinear stiffness Solver Parameters.");
  serac::mfem_ext::EquationSolver::DefineInputFileSchema(stiffness_solver_table);

  auto& dynamics_table = table.addTable("dynamics", "Parameters for mass matrix inversion");
  dynamics_table.addString("timestepper", "Timestepper (ODE) method to use");
  dynamics_table.addString("enforcement_method", "Time-varying constraint enforcement method to use");

  auto& bc_table = table.addGenericDictionary("boundary_conds", "Table of boundary conditions");
  serac::input::BoundaryConditionInputOptions::defineInputFileSchema(bc_table);

  auto& init_displ = table.addTable("initial_displacement", "Coefficient for initial condition");
  serac::input::CoefficientInputOptions::defineInputFileSchema(init_displ);
  auto& init_velo = table.addTable("initial_velocity", "Coefficient for initial condition");
  serac::input::CoefficientInputOptions::defineInputFileSchema(init_velo);
}

}  // namespace serac

using serac::DirichletEnforcementMethod;
using serac::NonlinearSolid;
using serac::TimestepMethod;

NonlinearSolid::InputOptions FromInlet<NonlinearSolid::InputOptions>::operator()(const axom::inlet::Table& base)
{
  NonlinearSolid::InputOptions result;

  result.order = base["order"];

  // Solver parameters
  auto stiffness_solver                  = base["stiffness_solver"];
  result.solver_options.H_lin_options    = stiffness_solver["linear"].get<serac::LinearSolverOptions>();
  result.solver_options.H_nonlin_options = stiffness_solver["nonlinear"].get<serac::NonlinearSolverOptions>();

  if (base.contains("dynamics")) {
    NonlinearSolid::TimesteppingOptions dyn_options;
    auto                                dynamics = base["dynamics"];

    // FIXME: Implement all supported methods as part of an ODE schema
    const static std::map<std::string, TimestepMethod> timestep_methods = {
        {"AverageAcceleration", TimestepMethod::AverageAcceleration}, {"NewmarkBeta", TimestepMethod::NewmarkBeta}};
    std::string timestep_method = dynamics["timestepper"];
    SLIC_ERROR_IF(timestep_methods.count(timestep_method) == 0, "Unrecognized timestep method: " << timestep_method);
    dyn_options.timestepper = timestep_methods.at(timestep_method);

    // FIXME: Implement all supported methods as part of an ODE schema
    const static std::map<std::string, DirichletEnforcementMethod> enforcement_methods = {
        {"RateControl", DirichletEnforcementMethod::RateControl}};
    std::string enforcement_method = dynamics["enforcement_method"];
    SLIC_ERROR_IF(enforcement_methods.count(enforcement_method) == 0,
                  "Unrecognized enforcement method: " << enforcement_method);
    dyn_options.enforcement_method = enforcement_methods.at(enforcement_method);

    result.solver_options.dyn_options = std::move(dyn_options);
  }

  // Set the material parameters
  // neo-Hookean material parameters
  result.mu = base["mu"];
  result.K  = base["K"];

  if (base.contains("boundary_conds")) {
    result.boundary_conditions =
        base["boundary_conds"].get<std::unordered_map<std::string, serac::input::BoundaryConditionInputOptions>>();
  }

  if (base.contains("initial_displacement")) {
    result.initial_displacement = base["initial_displacement"].get<serac::input::CoefficientInputOptions>();
  }
  if (base.contains("initial_velocity")) {
    result.initial_velocity = base["initial_velocity"].get<serac::input::CoefficientInputOptions>();
  }
  return result;
}

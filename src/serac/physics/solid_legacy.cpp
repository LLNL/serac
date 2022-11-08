// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_legacy.hpp"

#include "serac/infrastructure/logger.hpp"
#include "serac/physics/integrators/traction_integrator.hpp"
#include "serac/physics/integrators/displacement_hyperelastic_integrator.hpp"
#include "serac/physics/integrators/wrapper_integrator.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/numerics/expr_template_ops.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/coefficients/sensitivity_coefficients.hpp"

namespace serac {

/**
 * @brief The number of fields in this physics module (displacement and velocity)
 */
constexpr int NUM_FIELDS = 3;

SolidLegacy::SolidLegacy(int order, const SolverOptions& options, GeometricNonlinearities geom_nonlin,
                         FinalMeshOption keep_deformation, const std::string& name, mfem::ParMesh* pmesh)
    : BasePhysics(NUM_FIELDS, order, name, pmesh),
      velocity_(StateManager::newState(
          FiniteElementState::Options{
              .order = order, .vector_dim = mesh_.Dimension(), .name = detail::addPrefix(name, "velocity")},
          StateManager::collectionID(pmesh))),
      displacement_(StateManager::newState(
          FiniteElementState::Options{
              .order = order, .vector_dim = mesh_.Dimension(), .name = detail::addPrefix(name, "displacement")},
          StateManager::collectionID(pmesh))),
      adjoint_displacement_(StateManager::newState(
          FiniteElementState::Options{
              .order = order, .vector_dim = mesh_.Dimension(), .name = detail::addPrefix(name, "adjoint_displacement")},
          StateManager::collectionID(pmesh))),
      geom_nonlin_(geom_nonlin),
      keep_deformation_(keep_deformation),
      ode2_(displacement_.space().TrueVSize(),
            {.time = ode_time_point_, .c0 = c0_, .c1 = c1_, .u = u_, .du_dt = du_dt_, .d2u_dt2 = previous_},
            nonlin_solver_, bcs_)
{
  states_.push_back(velocity_);
  states_.push_back(displacement_);
  states_.push_back(adjoint_displacement_);

  // Initialize the mesh node pointers
  reference_nodes_ = std::make_unique<mfem::ParGridFunction>(&displacement_.space());
  mesh_.EnsureNodes();
  mesh_.GetNodes(*reference_nodes_);

  reference_nodes_->GetTrueDofs(x_);
  deformed_nodes_ = std::make_unique<mfem::ParGridFunction>(*reference_nodes_);

  if (!StateManager::isRestart()) {
    displacement_         = 0.0;
    velocity_             = 0.0;
    adjoint_displacement_ = 0.0;
  }

  const auto& lin_options = options.H_lin_options;
  // If the user wants the AMG preconditioner with a linear solver, set the pfes
  // to be the displacement
  const auto& augmented_options = mfem_ext::AugmentAMGForElasticity(lin_options, displacement_.space());

  nonlin_solver_ = mfem_ext::EquationSolver(mesh_.GetComm(), augmented_options, options.H_nonlin_options);

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

SolidLegacy::SolidLegacy(const SolidLegacy::InputOptions& options, const std::string& name)
    : SolidLegacy(options.order, options.solver_options, options.geom_nonlin, FinalMeshOption::Deformed, name)
{
  // This is the only other options stored in the input file that we can use
  // in the initialization stage
  setMaterialParameters(std::make_unique<mfem::ConstantCoefficient>(options.mu),
                        std::make_unique<mfem::ConstantCoefficient>(options.K), options.material_nonlin);

  auto dim = mesh_.Dimension();
  if (options.initial_displacement) {
    auto deform = options.initial_displacement->constructVector(dim);
    setDisplacement(*deform);
  }

  if (options.initial_velocity) {
    auto velo = options.initial_velocity->constructVector(dim);
    setVelocity(*velo);
  }
  setViscosity(std::make_unique<mfem::ConstantCoefficient>(options.viscosity));
  setMassDensity(std::make_unique<mfem::ConstantCoefficient>(options.initial_mass_density));

  for (const auto& [bc_name, bc] : options.boundary_conditions) {
    // FIXME: Better naming for boundary conditions?
    if (bc_name.find("displacement") != std::string::npos) {
      if (bc.coef_opts.isVector()) {
        std::shared_ptr<mfem::VectorCoefficient> disp_coef(bc.coef_opts.constructVector(dim));
        setDisplacementBCs(bc.attrs, disp_coef);
      } else {
        SLIC_ERROR_ROOT_IF(!bc.coef_opts.component,
                           "Component not specified with scalar coefficient when setting the displacement condition.");
        std::shared_ptr<mfem::Coefficient> disp_coef(bc.coef_opts.constructScalar());
        setDisplacementBCs(bc.attrs, disp_coef, *bc.coef_opts.component);
      }
    } else if (bc_name.find("traction") != std::string::npos) {
      std::shared_ptr<mfem::VectorCoefficient> trac_coef(bc.coef_opts.constructVector(dim));
      if (geom_nonlin_ == GeometricNonlinearities::Off) {
        setTractionBCs(bc.attrs, trac_coef, true);
      } else {
        setTractionBCs(bc.attrs, trac_coef, false);
      }
    } else if (bc_name.find("traction_ref") != std::string::npos) {
      std::shared_ptr<mfem::VectorCoefficient> trac_coef(bc.coef_opts.constructVector(dim));
      setTractionBCs(bc.attrs, trac_coef, true);
    } else if (bc_name.find("pressure") != std::string::npos) {
      std::shared_ptr<mfem::Coefficient> pres_coef(bc.coef_opts.constructScalar());
      if (geom_nonlin_ == GeometricNonlinearities::Off) {
        setPressureBCs(bc.attrs, pres_coef, true);
      } else {
        setPressureBCs(bc.attrs, pres_coef, false);
      }
    } else if (bc_name.find("pressure_ref") != std::string::npos) {
      std::shared_ptr<mfem::Coefficient> pres_coef(bc.coef_opts.constructScalar());
      setPressureBCs(bc.attrs, pres_coef, true);
    } else {
      SLIC_WARNING_ROOT("Ignoring boundary condition with unknown name: " << name);
    }
  }
}

SolidLegacy::~SolidLegacy()
{
  // Update the mesh with the new deformed nodes if requested
  if (keep_deformation_ == FinalMeshOption::Deformed && geom_nonlin_ == GeometricNonlinearities::On) {
    *reference_nodes_ += displacement_.gridFunction();
  }

  // Build a new grid function to store the mesh nodes post-destruction
  // NOTE: MFEM will manage the memory of these objects

  auto mesh_fe_coll  = new mfem::H1_FECollection(order_, mesh_.Dimension());
  auto mesh_fe_space = new mfem::ParFiniteElementSpace(displacement_.space(), &mesh_, mesh_fe_coll);
  auto mesh_nodes    = new mfem::ParGridFunction(mesh_fe_space);
  mesh_nodes->MakeOwner(mesh_fe_coll);

  *mesh_nodes = *reference_nodes_;

  // Set the mesh to the newly created nodes object and pass ownership
  mesh_.NewNodes(*mesh_nodes, true);
}

void SolidLegacy::setDisplacementBCs(const std::set<int>&                     disp_bdr,
                                     std::shared_ptr<mfem::VectorCoefficient> disp_bdr_coef)
{
  bcs_.addEssential(disp_bdr, disp_bdr_coef, displacement_.space());
}

void SolidLegacy::setDisplacementBCs(const std::set<int>& disp_bdr, std::shared_ptr<mfem::Coefficient> disp_bdr_coef,
                                     int component)
{
  bcs_.addEssential(disp_bdr, disp_bdr_coef, displacement_.space(), component);
}

void SolidLegacy::setTractionBCs(const std::set<int>& trac_bdr, std::shared_ptr<mfem::VectorCoefficient> trac_bdr_coef,
                                 bool compute_on_reference, std::optional<int> component)
{
  if (compute_on_reference) {
    bcs_.addGeneric(trac_bdr, trac_bdr_coef, SolidBoundaryCondition::ReferenceTraction, displacement_.space(),
                    component);
  } else {
    bcs_.addGeneric(trac_bdr, trac_bdr_coef, SolidBoundaryCondition::DeformedTraction, displacement_.space(),
                    component);
  }
}

void SolidLegacy::setPressureBCs(const std::set<int>& pres_bdr, std::shared_ptr<mfem::Coefficient> pres_bdr_coef,
                                 bool compute_on_reference)
{
  if (compute_on_reference) {
    bcs_.addGeneric(pres_bdr, pres_bdr_coef, SolidBoundaryCondition::ReferencePressure, displacement_.space());
  } else {
    bcs_.addGeneric(pres_bdr, pres_bdr_coef, SolidBoundaryCondition::DeformedPressure, displacement_.space());
  }
}

void SolidLegacy::addBodyForce(std::shared_ptr<mfem::VectorCoefficient> ext_force_coef)
{
  ext_force_coefs_.push_back(ext_force_coef);
}

void SolidLegacy::setMaterialParameters(std::unique_ptr<mfem::Coefficient>&& mu, std::unique_ptr<mfem::Coefficient>&& K,
                                        const bool material_nonlin)
{
  if (material_nonlin) {
    material_ = std::make_unique<NeoHookeanMaterial>(std::move(mu), std::move(K));
  } else {
    material_ = std::make_unique<LinearElasticMaterial>(std::move(mu), std::move(K));
  }
}

void SolidLegacy::setThermalExpansion(std::unique_ptr<mfem::Coefficient>&& coef_thermal_expansion,
                                      std::unique_ptr<mfem::Coefficient>&& reference_temp,
                                      const FiniteElementState&            temp)
{
  thermal_material_ = std::make_unique<IsotropicThermalExpansionMaterial>(
      std::move(coef_thermal_expansion), std::move(reference_temp), temp, geom_nonlin_);
}

void SolidLegacy::setViscosity(std::unique_ptr<mfem::Coefficient>&& visc_coef) { viscosity_ = std::move(visc_coef); }

void SolidLegacy::setMassDensity(std::unique_ptr<mfem::Coefficient>&& rho_coef)
{
  initial_mass_density_ = std::move(rho_coef);
}

void SolidLegacy::setDisplacement(mfem::VectorCoefficient& disp_state)
{
  disp_state.SetTime(time_);
  displacement_.project(disp_state);
  gf_initialized_[1] = true;
}

void SolidLegacy::setVelocity(mfem::VectorCoefficient& velo_state)
{
  velo_state.SetTime(time_);
  velocity_.project(velo_state);
  gf_initialized_[0] = true;
}

void SolidLegacy::resetToReferenceConfiguration()
{
  displacement_ = 0.0;
  velocity_     = 0.0;

  mesh_.NewNodes(*reference_nodes_);
}

void SolidLegacy::completeSetup()
{
  // Define the nonlinear form
  H_ = std::make_unique<mfem::ParNonlinearForm>(&displacement_.space());

  // Add the hyperelastic integrator
  H_->AddDomainIntegrator(
      new mfem_ext::DisplacementHyperelasticIntegrator(*material_, *thermal_material_, geom_nonlin_));

  // Add the deformed traction integrator
  for (auto& deformed_traction_data : bcs_.genericsWithTag(SolidBoundaryCondition::DeformedTraction)) {
    H_->AddBdrFaceIntegrator(new mfem_ext::TractionIntegrator(deformed_traction_data.vectorCoefficient(), false),
                             deformed_traction_data.markers());
  }

  // Add the reference traction integrator
  for (auto& deformed_traction_data : bcs_.genericsWithTag(SolidBoundaryCondition::ReferenceTraction)) {
    H_->AddBdrFaceIntegrator(new mfem_ext::TractionIntegrator(deformed_traction_data.vectorCoefficient(), true),
                             deformed_traction_data.markers());
  }

  // Add the deformed pressure integrator
  for (auto& deformed_pressure_data : bcs_.genericsWithTag(SolidBoundaryCondition::DeformedPressure)) {
    H_->AddBdrFaceIntegrator(new mfem_ext::PressureIntegrator(deformed_pressure_data.scalarCoefficient(), false),
                             deformed_pressure_data.markers());
  }

  // Add the reference pressure integrator
  for (auto& reference_pressure_data : bcs_.genericsWithTag(SolidBoundaryCondition::ReferencePressure)) {
    H_->AddBdrFaceIntegrator(new mfem_ext::PressureIntegrator(reference_pressure_data.scalarCoefficient(), true),
                             reference_pressure_data.markers());
  }

  // Add external forces
  for (auto& force : ext_force_coefs_) {
    H_->AddDomainIntegrator(new serac::mfem_ext::LinearToNonlinearFormIntegrator(
        std::make_shared<mfem::VectorDomainLFIntegrator>(*force), displacement_.space()));
  }

  // Build the dof array lookup tables
  displacement_.space().BuildDofToArrays();

  // If dynamic, create the mass and viscosity forms
  if (!is_quasistatic_) {
    M_ = std::make_unique<mfem::ParBilinearForm>(&displacement_.space());
    M_->AddDomainIntegrator(new mfem::VectorMassIntegrator(*initial_mass_density_));
    M_->Assemble(0);
    M_->Finalize(0);

    M_mat_.reset(M_->ParallelAssemble());

    C_ = std::make_unique<mfem::ParBilinearForm>(&displacement_.space());
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
          r = (*M_mat_) * d2u_dt2 + (*C_mat_) * (du_dt_ + c1_ * d2u_dt2) + (*H_) * (u_ + c0_ * d2u_dt2);
          r.SetSubVector(bcs_.allEssentialTrueDofs(), 0.0);
        },

        // gradient of residual function
        [this](const mfem::Vector& d2u_dt2) -> mfem::Operator& {
          // J = M + c1 * C + c0 * H(u_predicted)
          auto localJ = std::unique_ptr<mfem::SparseMatrix>(Add(1.0, M_->SpMat(), c1_, C_->SpMat()));
          localJ->Add(c0_, H_->GetLocalGradient(u_ + c0_ * d2u_dt2));
          J_mat_.reset(M_->ParallelAssemble(localJ.get()));
          bcs_.eliminateAllEssentialDofsFromMatrix(*J_mat_);
          return *J_mat_;
        });
  }

  nonlin_solver_.SetOperator(*residual_);
}

// Solve the Quasi-static Newton system
void SolidLegacy::quasiStaticSolve() { nonlin_solver_.Mult(zero_, displacement_); }

std::unique_ptr<mfem::Operator> SolidLegacy::buildQuasistaticOperator()
{
  // the quasistatic case is entirely described by the residual,
  // there is no ordinary differential equation
  auto residual = std::make_unique<mfem_ext::StdFunctionOperator>(
      displacement_.space().TrueVSize(),

      // residual function
      [this](const mfem::Vector& u, mfem::Vector& r) {
        H_->Mult(u, r);  // r := H(u)
        r.SetSubVector(bcs_.allEssentialTrueDofs(), 0.0);
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
void SolidLegacy::advanceTimestep(double& dt)
{
  // Set the mesh nodes to the reference configuration
  if (geom_nonlin_ == GeometricNonlinearities::On) {
    mesh_.NewNodes(*reference_nodes_);
  }

  // If a thermal material is present, evaluate the grid function
  auto* iso_expansion_mat = dynamic_cast<IsotropicThermalExpansionMaterial*>(thermal_material_.get());
  if (iso_expansion_mat) {
    iso_expansion_mat->updateGridFunction();
  }

  if (is_quasistatic_) {
    // Update the time for housekeeping purposes
    time_ += dt;
    // Project the essential boundary coefficients
    for (auto& bc : bcs_.essentials()) {
      bc.setDofs(displacement_, time_);
    }

    quasiStaticSolve();
  } else {
    // Note that the ODE solver handles the essential boundary condition application itself
    ode2_.Step(displacement_, velocity_, time_, dt);
  }

  if (geom_nonlin_ == GeometricNonlinearities::On) {
    // Update the mesh with the new deformed nodes
    deformed_nodes_->Set(1.0, displacement_.gridFunction());
    deformed_nodes_->Add(1.0, *reference_nodes_);
    mesh_.NewNodes(*deformed_nodes_);
  }

  cycle_ += 1;

  previous_solve_ = PreviousSolve::Forward;
}

void SolidLegacy::checkSensitivityMode() const
{
  SLIC_ERROR_ROOT_IF(previous_solve_ == PreviousSolve::None,
                     "Sensitivities only valid following a forward and adjoint solve.");
  SLIC_WARNING_ROOT_IF(
      previous_solve_ == PreviousSolve::Forward,
      "Sensitivities only valid following a forward and adjoint solve (in that order). The previous solve was a "
      "forward analysis. Ensure that the correct displacement and adjoint states are set for sensitivies.");

  LinearElasticMaterial* linear_mat = dynamic_cast<LinearElasticMaterial*>(material_.get());

  SLIC_ERROR_ROOT_IF(!linear_mat, "Only linear elastic materials allowed for sensitivity analysis.");
}

FiniteElementDual& SolidLegacy::shearModulusSensitivity(mfem::ParFiniteElementSpace* shear_space)
{
  checkSensitivityMode();

  if (geom_nonlin_ == GeometricNonlinearities::On) {
    // Set the mesh nodes to the reference configuration
    mesh_.NewNodes(*reference_nodes_);
  }

  LinearElasticMaterial* linear_mat = dynamic_cast<LinearElasticMaterial*>(material_.get());

  shear_sensitivity_coef_ =
      std::make_unique<mfem_ext::ShearSensitivityCoefficient>(displacement_, adjoint_displacement_, *linear_mat);

  // Add a scalar linear form integrator using the shear sensitivity coefficient against the given shear modulus finite
  // element space
  if (!shear_sensitivity_form_ || shear_space) {
    SLIC_ERROR_IF(!shear_space,
                  axom::fmt::format("Finite element space is required for first shear sensitivity call."));
    shear_sensitivity_      = std::make_unique<FiniteElementDual>(*shear_space);
    shear_sensitivity_form_ = std::make_unique<mfem::ParLinearForm>(&shear_sensitivity_->space());

    shear_sensitivity_form_->AddDomainIntegrator(new mfem::DomainLFIntegrator(*shear_sensitivity_coef_, 2, 2));
  }

  // Assemble the linear form at the current state and adjoint values
  shear_sensitivity_form_->Assemble();

  // Set the dual state to the assembled shear sensitivity
  std::unique_ptr<mfem::HypreParVector> assembled_vec(shear_sensitivity_form_->ParallelAssemble());
  auto* shear_sensitivity_vector = dynamic_cast<mfem::HypreParVector*>(shear_sensitivity_.get());
  *shear_sensitivity_vector      = *assembled_vec;

  if (geom_nonlin_ == GeometricNonlinearities::On) {
    // Set the mesh nodes back to the reference configuration
    mesh_.NewNodes(*deformed_nodes_);
  }

  return *shear_sensitivity_;
}

FiniteElementDual& SolidLegacy::bulkModulusSensitivity(mfem::ParFiniteElementSpace* bulk_space)
{
  checkSensitivityMode();

  if (geom_nonlin_ == GeometricNonlinearities::On) {
    // Set the mesh nodes to the reference configuration
    mesh_.NewNodes(*reference_nodes_);
  }

  LinearElasticMaterial* linear_mat = dynamic_cast<LinearElasticMaterial*>(material_.get());

  bulk_sensitivity_coef_ =
      std::make_unique<mfem_ext::BulkSensitivityCoefficient>(displacement_, adjoint_displacement_, *linear_mat);

  // Add a scalar linear form integrator using the shear sensitivity coefficient against the given bulk modulus finite
  // element space
  if (!bulk_sensitivity_form_ || bulk_space) {
    SLIC_ERROR_IF(!bulk_space, axom::fmt::format("Finite element space is required for first bulk sensitivity call."));
    bulk_sensitivity_      = std::make_unique<FiniteElementDual>(*bulk_space);
    bulk_sensitivity_form_ = std::make_unique<mfem::ParLinearForm>(&bulk_sensitivity_->space());

    bulk_sensitivity_form_->AddDomainIntegrator(new mfem::DomainLFIntegrator(*bulk_sensitivity_coef_, 2, 2));
  }

  // Assemble the linear form at the current state and adjoint values
  bulk_sensitivity_form_->Assemble();

  // Set the dual state to the assembled bulk sensitivity
  std::unique_ptr<mfem::HypreParVector> assembled_vec(bulk_sensitivity_form_->ParallelAssemble());
  auto* bulk_sensitivity_vector = dynamic_cast<mfem::HypreParVector*>(bulk_sensitivity_.get());
  *bulk_sensitivity_vector      = *assembled_vec;

  if (geom_nonlin_ == GeometricNonlinearities::On) {
    // Set the mesh nodes back to the reference configuration
    mesh_.NewNodes(*deformed_nodes_);
  }

  return *bulk_sensitivity_;
}

const FiniteElementState& SolidLegacy::solveAdjoint(FiniteElementDual& adjoint_load,
                                                    FiniteElementDual* dual_with_essential_boundary)
{
  SLIC_ERROR_ROOT_IF(!is_quasistatic_, "Adjoint analysis only vaild for quasistatic problems.");
  SLIC_ERROR_ROOT_IF(previous_solve_ == PreviousSolve::None, "Adjoint analysis only valid following a forward solve.");

  if (geom_nonlin_ == GeometricNonlinearities::On) {
    // Set the mesh nodes to the reference configuration
    mesh_.NewNodes(*reference_nodes_);
  }

  // note: The assignment operator must be called after the copy constructor because
  // the copy constructor only sets the partitioning, it does not copy the actual vector
  // values
  mfem::HypreParVector adjoint_load_vector(adjoint_load);

  auto& lin_solver = nonlin_solver_.LinearSolver();

  auto& J   = dynamic_cast<mfem::HypreParMatrix&>(H_->GetGradient(displacement_));
  auto  J_T = std::unique_ptr<mfem::HypreParMatrix>(J.Transpose());

  // By default, use a homogeneous essential boundary condition
  mfem::HypreParVector adjoint_essential(adjoint_load);
  adjoint_essential = 0.0;

  // If we have a non-homogeneous essential boundary condition, extract it from the given state
  if (dual_with_essential_boundary) {
    adjoint_essential = *dual_with_essential_boundary;
  }

  for (const auto& bc : bcs_.essentials()) {
    bc.apply(*J_T, adjoint_load_vector, adjoint_essential);
  }

  lin_solver.SetOperator(*J_T);
  lin_solver.Mult(adjoint_load_vector, adjoint_displacement_);

  if (geom_nonlin_ == GeometricNonlinearities::On) {
    mesh_.NewNodes(*deformed_nodes_);
  }

  // Reset the equation solver to use the full nonlinear residual operator
  nonlin_solver_.SetOperator(*residual_);

  previous_solve_ = PreviousSolve::Adjoint;

  return adjoint_displacement_;
}

void SolidLegacy::InputOptions::defineInputFileSchema(axom::inlet::Container& container)
{
  // Polynomial interpolation order - currently up to 8th order is allowed
  container.addInt("order", "Order degree of the finite elements.").defaultValue(1).range(1, 8);

  // neo-Hookean material parameters
  container.addDouble("mu", "Shear modulus in the Neo-Hookean hyperelastic model.").defaultValue(0.25);
  container.addDouble("K", "Bulk modulus in the Neo-Hookean hyperelastic model.").defaultValue(5.0);

  // Geometric nonlinearities flag
  container.addBool("geometric_nonlin", "Flag to include geometric nonlinearities in the residual calculation.")
      .defaultValue(true);

  // Geometric nonlinearities flag
  container
      .addBool("material_nonlin",
               "Flag to include material nonlinearities (linear elastic vs. neo-Hookean material model).")
      .defaultValue(true);

  container.addDouble("viscosity", "Viscosity constant").defaultValue(0.0);

  container.addDouble("density", "Initial mass density").defaultValue(1.0);

  auto& equation_solver_container =
      container.addStruct("equation_solver", "Linear and Nonlinear stiffness Solver Parameters.");
  serac::mfem_ext::EquationSolver::DefineInputFileSchema(equation_solver_container);

  auto& dynamics_container = container.addStruct("dynamics", "Parameters for mass matrix inversion");
  dynamics_container.addString("timestepper", "Timestepper (ODE) method to use");
  dynamics_container.addString("enforcement_method", "Time-varying constraint enforcement method to use");

  auto& bc_container = container.addStructDictionary("boundary_conds", "Container of boundary conditions");
  serac::input::BoundaryConditionInputOptions::defineInputFileSchema(bc_container);

  auto& init_displ = container.addStruct("initial_displacement", "Coefficient for initial condition");
  serac::input::CoefficientInputOptions::defineInputFileSchema(init_displ);
  auto& init_velo = container.addStruct("initial_velocity", "Coefficient for initial condition");
  serac::input::CoefficientInputOptions::defineInputFileSchema(init_velo);
}

// Evaluate the residual at the current state
mfem::Vector SolidLegacy::currentResidual()
{
  mfem::Vector eval(displacement_.Size());
  if (is_quasistatic_) {
    // The input to the residual is displacment
    residual_->Mult(displacement_, eval);
  } else {
    // Currently the residual constructed uses d2u_dt2 as input,
    // but this could change
    residual_->Mult(ode2_.GetState().d2u_dt2, eval);
  }
  return eval;
}

// Get an Operator that computes the gradient (tangent stiffness) at the current internal state
const mfem::Operator& SolidLegacy::currentGradient()
{
  if (is_quasistatic_) {
    // The input to the residual is displacment
    return residual_->GetGradient(displacement_);
  }

  // Currently the residual constructed uses d2u_dt2 as input,
  // but this could change
  return residual_->GetGradient(ode2_.GetState().d2u_dt2);
}

}  // namespace serac

using serac::DirichletEnforcementMethod;
using serac::SolidLegacy;
using serac::TimestepMethod;

serac::SolidLegacy::InputOptions FromInlet<serac::SolidLegacy::InputOptions>::operator()(
    const axom::inlet::Container& base)
{
  SolidLegacy::InputOptions result;

  result.order = base["order"];

  // Solver parameters
  auto equation_solver                   = base["equation_solver"];
  result.solver_options.H_lin_options    = equation_solver["linear"].get<serac::LinearSolverOptions>();
  result.solver_options.H_nonlin_options = equation_solver["nonlinear"].get<serac::NonlinearSolverOptions>();

  if (base.contains("dynamics")) {
    SolidLegacy::TimesteppingOptions dyn_options;
    auto                             dynamics = base["dynamics"];

    // FIXME: Implement all supported methods as part of an ODE schema
    const static std::map<std::string, TimestepMethod> timestep_methods = {
        {"AverageAcceleration", TimestepMethod::AverageAcceleration},
        {"NewmarkBeta", TimestepMethod::Newmark},
        {"BackwardEuler", TimestepMethod::BackwardEuler}};
    std::string timestep_method = dynamics["timestepper"];
    SLIC_ERROR_ROOT_IF(timestep_methods.count(timestep_method) == 0,
                       "Unrecognized timestep method: " << timestep_method);
    dyn_options.timestepper = timestep_methods.at(timestep_method);

    // FIXME: Implement all supported methods as part of an ODE schema
    const static std::map<std::string, DirichletEnforcementMethod> enforcement_methods = {
        {"RateControl", DirichletEnforcementMethod::RateControl}};
    std::string enforcement_method = dynamics["enforcement_method"];
    SLIC_ERROR_ROOT_IF(enforcement_methods.count(enforcement_method) == 0,
                       "Unrecognized enforcement method: " << enforcement_method);
    dyn_options.enforcement_method = enforcement_methods.at(enforcement_method);

    result.solver_options.dyn_options = std::move(dyn_options);
  }

  // Set the material parameters
  // neo-Hookean material parameters
  result.mu = base["mu"];
  result.K  = base["K"];

  // Set the geometric nonlinearities flag
  bool input_geom_nonlin = base["geometric_nonlin"];
  if (input_geom_nonlin) {
    result.geom_nonlin = serac::GeometricNonlinearities::On;
  } else {
    result.geom_nonlin = serac::GeometricNonlinearities::Off;
  }

  // Set the material nonlinearity flag
  result.material_nonlin = base["material_nonlin"];

  if (base.contains("boundary_conds")) {
    result.boundary_conditions =
        base["boundary_conds"].get<std::unordered_map<std::string, serac::input::BoundaryConditionInputOptions>>();
  }

  result.viscosity = base["viscosity"];

  result.initial_mass_density = base["density"];

  if (base.contains("initial_displacement")) {
    result.initial_displacement = base["initial_displacement"].get<serac::input::CoefficientInputOptions>();
  }
  if (base.contains("initial_velocity")) {
    result.initial_velocity = base["initial_velocity"].get<serac::input::CoefficientInputOptions>();
  }
  return result;
}

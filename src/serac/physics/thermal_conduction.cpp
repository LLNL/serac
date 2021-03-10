// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/thermal_conduction.hpp"

#include "serac/infrastructure/logger.hpp"
#include "serac/numerics/expr_template_ops.hpp"
#include "serac/physics/integrators/nonlinear_reaction_integrator.hpp"
#include "serac/physics/integrators/wrapper_integrator.hpp"

namespace serac {

constexpr int NUM_FIELDS = 1;

ThermalConduction::ThermalConduction(int order, const SolverOptions& options, const std::string& name)
    : BasePhysics(NUM_FIELDS, order),
      temperature_(StateManager::newState(FiniteElementState::Options{.order      = order,
                                                                      .vector_dim = 1,
                                                                      .ordering   = mfem::Ordering::byNODES,
                                                                      .name = detail::addPrefix(name, "temperature")})),
      residual_(temperature_.space().TrueVSize()),
      ode_(temperature_.space().TrueVSize(), {.u = u_, .dt = dt_, .du_dt = previous_, .previous_dt = previous_dt_},
           nonlin_solver_, bcs_)
{
  state_.push_back(temperature_);

  nonlin_solver_ = mfem_ext::EquationSolver(mesh_.GetComm(), options.T_lin_options, options.T_nonlin_options);
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

ThermalConduction::ThermalConduction(const InputOptions& options)
    : ThermalConduction(options.order, options.solver_options)
{
  setConductivity(std::make_unique<mfem::ConstantCoefficient>(options.kappa));
  setMassDensity(std::make_unique<mfem::ConstantCoefficient>(options.rho));
  setSpecificHeatCapacity(std::make_unique<mfem::ConstantCoefficient>(options.cp));

  if (options.reaction_func) {
    if (options.reaction_scale_coef) {
      setNonlinearReaction(options.reaction_func, options.d_reaction_func,
                           options.reaction_scale_coef->constructScalar());
    } else {
      setNonlinearReaction(options.reaction_func, options.d_reaction_func,
                           std::make_unique<mfem::ConstantCoefficient>(1.0));
    }
  }

  if (options.initial_temperature) {
    auto temp = options.initial_temperature->constructScalar();
    setTemperature(*temp);
  }

  if (options.source_coef) {
    setSource(options.source_coef->constructScalar());
  }

  // Process the BCs in sorted order for correct behavior with repeated attributes
  std::map<std::string, input::BoundaryConditionInputOptions> sorted_bcs(options.boundary_conditions.begin(),
                                                                         options.boundary_conditions.end());
  for (const auto& [name, bc] : sorted_bcs) {
    // FIXME: Better naming for boundary conditions?
    if (name.find("temperature") != std::string::npos) {
      std::shared_ptr<mfem::Coefficient> temp_coef(bc.coef_opts.constructScalar());
      setTemperatureBCs(bc.attrs, temp_coef);
    } else if (name.find("flux") != std::string::npos) {
      std::shared_ptr<mfem::Coefficient> flux_coef(bc.coef_opts.constructScalar());
      setFluxBCs(bc.attrs, flux_coef);
    } else {
      SLIC_WARNING_ROOT("Ignoring boundary condition with unknown name: " << name);
    }
  }
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

void ThermalConduction::setNonlinearReaction(std::function<double(double)>        reaction,
                                             std::function<double(double)>        d_reaction,
                                             std::unique_ptr<mfem::Coefficient>&& scale)
{
  reaction_       = reaction;
  d_reaction_     = d_reaction;
  reaction_scale_ = std::move(scale);
}

void ThermalConduction::setSpecificHeatCapacity(std::unique_ptr<mfem::Coefficient>&& cp)
{
  // Set the specific heat capacity coefficient
  cp_ = std::move(cp);
}

void ThermalConduction::setMassDensity(std::unique_ptr<mfem::Coefficient>&& rho)
{
  // Set the density coefficient
  rho_ = std::move(rho);
}

void ThermalConduction::completeSetup()
{
  SLIC_ASSERT_MSG(kappa_, "Conductivity not set in ThermalSolver!");

  // Add the domain diffusion integrator to the K form
  K_form_ = temperature_.createOnSpace<mfem::ParNonlinearForm>();
  K_form_->AddDomainIntegrator(
      new mfem_ext::BilinearToNonlinearFormIntegrator(std::make_unique<mfem::DiffusionIntegrator>(*kappa_)));

  // Add the body source to the RS if specified
  if (source_) {
    K_form_->AddDomainIntegrator(new mfem_ext::LinearToNonlinearFormIntegrator(
        std::make_unique<mfem::DomainLFIntegrator>(*source_), temperature_.space()));
  }

  // Add a nonlinear reaction term if specified
  if (reaction_) {
    K_form_->AddDomainIntegrator(
        new serac::mfem_ext::NonlinearReactionIntegrator(reaction_, d_reaction_, *reaction_scale_));
  }

  // Build the dof array lookup tables
  temperature_.space().BuildDofToArrays();

  // Project the essential boundary coefficients
  for (auto& bc : bcs_.essentials()) {
    bc.projectBdr(temperature_, time_);
  }

  // Initialize the true vector
  temperature_.initializeTrueVec();

  if (is_quasistatic_) {
    residual_ = mfem_ext::StdFunctionOperator(
        temperature_.space().TrueVSize(),

        [this](const mfem::Vector& u, mfem::Vector& r) {
          K_form_->Mult(u, r);
          r.SetSubVector(bcs_.allEssentialDofs(), 0.0);
        },

        [this](const mfem::Vector& u) -> mfem::Operator& {
          auto& J = dynamic_cast<mfem::HypreParMatrix&>(K_form_->GetGradient(u));
          bcs_.eliminateAllEssentialDofsFromMatrix(J);
          return J;
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
          r = (*M_) * du_dt + (*K_form_) * (u_ + dt_ * du_dt);
          r.SetSubVector(bcs_.allEssentialDofs(), 0.0);
        },

        [this](const mfem::Vector& du_dt) -> mfem::Operator& {
          // Only reassemble the stiffness if it is a new timestep or we have a nonlinear reaction
          if (dt_ != previous_dt_ || reaction_) {
            auto localJ = std::unique_ptr<mfem::SparseMatrix>(
                mfem::Add(1.0, M_form_->SpMat(), dt_, K_form_->GetLocalGradient(u_ + dt_ * du_dt)));
            J_.reset(M_form_->ParallelAssemble(localJ.get()));
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

void ThermalConduction::InputOptions::defineInputFileSchema(axom::inlet::Table& table)
{
  // Polynomial interpolation order - currently up to 8th order is allowed
  table.addInt("order", "Order degree of the finite elements.").defaultValue(1).range(1, 8);

  // material parameters
  table.addDouble("kappa", "Thermal conductivity").defaultValue(0.5);
  table.addDouble("rho", "Density").defaultValue(1.0);
  table.addDouble("cp", "Specific heat capacity").defaultValue(1.0);

  auto& source = table.addStruct("source", "Scalar source term (RHS of the thermal conduction PDE)");
  serac::input::CoefficientInputOptions::defineInputFileSchema(source);

  auto& reaction_table = table.addStruct("nonlinear_reaction", "Nonlinear reaction term parameters");
  reaction_table.addFunction("reaction_function", axom::inlet::FunctionTag::Double, {axom::inlet::FunctionTag::Double},
                             "Nonlinear reaction function q = q(temperature)");
  reaction_table.addFunction("d_reaction_function", axom::inlet::FunctionTag::Double,
                             {axom::inlet::FunctionTag::Double},
                             "Derivative of the nonlinear reaction function dq = dq / dTemperature");
  auto& scale_coef_table = reaction_table.addStruct("scale", "Spatially varying scale factor for the reaction");
  serac::input::CoefficientInputOptions::defineInputFileSchema(scale_coef_table);

  auto& equation_solver_table = table.addStruct("equation_solver", "Linear and Nonlinear stiffness Solver Parameters.");
  serac::mfem_ext::EquationSolver::DefineInputFileSchema(equation_solver_table);

  auto& dynamics_table = table.addStruct("dynamics", "Parameters for mass matrix inversion");
  dynamics_table.addString("timestepper", "Timestepper (ODE) method to use");
  dynamics_table.addString("enforcement_method", "Time-varying constraint enforcement method to use");

  auto& bc_table = table.addStructDictionary("boundary_conds", "Table of boundary conditions");
  serac::input::BoundaryConditionInputOptions::defineInputFileSchema(bc_table);

  auto& init_temp = table.addStruct("initial_temperature", "Coefficient for initial condition");
  serac::input::CoefficientInputOptions::defineInputFileSchema(init_temp);
}

}  // namespace serac

using serac::DirichletEnforcementMethod;
using serac::ThermalConduction;
using serac::TimestepMethod;

ThermalConduction::InputOptions FromInlet<ThermalConduction::InputOptions>::operator()(const axom::inlet::Table& base)
{
  ThermalConduction::InputOptions result;

  result.order = base["order"];

  // Solver parameters
  auto equation_solver                   = base["equation_solver"];
  result.solver_options.T_lin_options    = equation_solver["linear"].get<serac::LinearSolverOptions>();
  result.solver_options.T_nonlin_options = equation_solver["nonlinear"].get<serac::NonlinearSolverOptions>();

  if (base.contains("dynamics")) {
    ThermalConduction::TimesteppingOptions dyn_options;
    auto                                   dynamics = base["dynamics"];

    // FIXME: Implement all supported methods as part of an ODE schema
    const static std::map<std::string, TimestepMethod> timestep_methods = {
        {"AverageAcceleration", TimestepMethod::AverageAcceleration},
        {"BackwardEuler", TimestepMethod::BackwardEuler},
        {"ForwardEuler", TimestepMethod::ForwardEuler}};
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

  if (base.contains("nonlinear_reaction")) {
    auto reaction          = base["nonlinear_reaction"];
    result.reaction_func   = reaction["reaction_function"].get<std::function<double(double)>>();
    result.d_reaction_func = reaction["d_reaction_function"].get<std::function<double(double)>>();
    if (reaction.contains("scale")) {
      result.reaction_scale_coef = reaction["scale"].get<serac::input::CoefficientInputOptions>();
    }
  }

  if (base.contains("source")) {
    result.source_coef = base["source"].get<serac::input::CoefficientInputOptions>();
  }

  // Set the material parameters
  result.kappa = base["kappa"];
  result.rho   = base["rho"];
  result.cp    = base["cp"];

  result.boundary_conditions =
      base["boundary_conds"].get<std::unordered_map<std::string, serac::input::BoundaryConditionInputOptions>>();

  if (base.contains("initial_temperature")) {
    result.initial_temperature = base["initial_temperature"].get<serac::input::CoefficientInputOptions>();
  }
  return result;
}

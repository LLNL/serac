// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/thermal_conduction_functional.hpp"

#include "serac/infrastructure/logger.hpp"
#include "serac/numerics/expr_template_ops.hpp"
#include "serac/physics/integrators/nonlinear_reaction_integrator.hpp"
#include "serac/physics/integrators/wrapper_integrator.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/numerics/functional/detail/metaprogramming.hpp"

namespace serac {

constexpr int NUM_FIELDS = 1;

constexpr int MAX_ORDER = 4;

template<int order, int dim>
ThermalConductionFunctional<order, dim>::ThermalConductionFunctional(const SolverOptions& options, const std::string& name)
    : BasePhysics(NUM_FIELDS, order),
      temperature_(StateManager::newState(FiniteElementState::Options{.order      = order,
                                                                      .vector_dim = 1,
                                                                      .ordering   = mfem::Ordering::byNODES,
                                                                      .name = detail::addPrefix(name, "temperature")})),
      K_functional_(&temperature_.space(), &temperature_.space()),
      M_functional_(&temperature_.space(), &temperature_.space()),
      residual_(temperature_.space().TrueVSize()),
      ode_(temperature_.space().TrueVSize(), {.u = u_, .dt = dt_, .du_dt = previous_, .previous_dt = previous_dt_},
           nonlin_solver_, bcs_)
{
  static_assert(order > 0 && order < MAX_ORDER, "Invalid order requested in the thermal conduction module");

  SLIC_ERROR_ROOT_IF(mesh_.Dimension() != dim,
                     fmt::format("Compile time dimension and runtime mesh dimension mismatch"));

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
  cp_  = [](const mfem::Vector& /* x */, double /* t */) { return 1.0; };
  rho_ = [](const mfem::Vector& /* x */, double /* t */) { return 1.0; };
}

template<int order, int dim>
ThermalConductionFunctional<order, dim>::ThermalConductionFunctional(const InputOptions& options, const std::string& name)
    : ThermalConductionFunctional<order, dim>(options.solver_options, name)
{
  setConductivity([kappa = options.kappa](const mfem::Vector&, double) { return kappa; });
  setMassDensity([rho = options.rho](const mfem::Vector&, double) { return rho; });
  setSpecificHeatCapacity([cp = options.cp](const mfem::Vector&, double) { return cp; });

  if (options.initial_temperature_func) {
    setTemperature(options.initial_temperature_func->scalar_function);
  }

  if (options.source_func) {
    setSource(options.source_func->scalar_function);
  }

  // Process the BCs in sorted order for correct behavior with repeated attributes
  std::map<std::string, input::BoundaryConditionInputOptions> sorted_bcs(options.boundary_conditions.begin(),
                                                                         options.boundary_conditions.end());
  for (const auto& [bc_name, bc] : sorted_bcs) {
    // FIXME: Better naming for boundary conditions?
    if (bc_name.find("temperature") != std::string::npos) {
      std::shared_ptr<mfem::Coefficient> temp_coef(bc.coef_opts.constructScalar());
      setTemperatureBCs(bc.attrs, temp_coef);
    } else if (bc_name.find("flux") != std::string::npos) {
      std::shared_ptr<mfem::Coefficient> flux_coef(bc.coef_opts.constructScalar());
      setFluxBCs(bc.attrs, flux_coef);
    } else {
      SLIC_WARNING_ROOT("Ignoring boundary condition with unknown name: " << name);
    }
  }
}

template<int order, int dim>
void ThermalConductionFunctional<order, dim>::setTemperature(ScalarFunc temp)
{
  // Project the coefficient onto the grid function
  mfem::FunctionCoefficient temp_coef(temp);

  temp_coef.SetTime(time_);
  temperature_.project(temp_coef);
  gf_initialized_[0] = true;
}

template<int order, int dim>
void ThermalConductionFunctional<order, dim>::setTemperatureBCs(const std::set<int>& temp_bdr,
                                          std::shared_ptr<mfem::Coefficient> temp_bdr_coef)
{
  bcs_.addEssential(temp_bdr, temp_bdr_coef, temperature_);
}

template<int order, int dim>
void ThermalConductionFunctional<order, dim>::setFluxBCs(const std::set<int>& flux_bdr, std::shared_ptr<mfem::Coefficient> flux_bdr_coef)
{
  // Set the natural (integral) boundary condition
  bcs_.addNatural(flux_bdr, flux_bdr_coef, -1);
}

template<int order, int dim>
void ThermalConductionFunctional<order, dim>::setConductivity(ScalarFunc kappa)
{
  // Set the conduction coefficient
  kappa_ = std::move(kappa);
}

template<int order, int dim>
void ThermalConductionFunctional<order, dim>::setSource(ScalarFunc source)
{
  // Set the body source integral coefficient
  source_ = std::move(source);
}

template<int order, int dim>
void ThermalConductionFunctional<order, dim>::setSpecificHeatCapacity(ScalarFunc cp)
{
  // Set the specific heat capacity coefficient
  cp_ = std::move(cp);
}

template<int order, int dim>
void ThermalConductionFunctional<order, dim>::setMassDensity(ScalarFunc rho)
{
  // Set the density coefficient
  rho_ = std::move(rho);
}

template<int order, int dim>
void ThermalConductionFunctional<order, dim>::completeSetup()
{
  SLIC_ASSERT_MSG(kappa_, "Conductivity not set in ThermalSolver!");

  K_functional_.AddDomainIntegral(
      Dimension<dim>{},
      [this]([[maybe_unused]] auto x, auto temperature) {
        // Get the value and the gradient from the input tuple
        auto [u, du_dx] = temperature;

        auto source = zero{};

        auto flux = kappa_(x, t_) * du_dx;

        // Return the source and the flux as a tuple 
        return serac::tuple{source, flux};
      },
      mesh_);

  // Build the dof array lookup tables
  temperature_.space().BuildDofToArrays();

  // Project the essential boundary coefficients
  for (auto& bc : bcs_.essentials()) {
    bc.projectBdr(temperature_, time_);
    K_functional_.AddEssentialBC(bc.markers());
  }

  // Initialize the true vector
  temperature_.initializeTrueVec();

  if (is_quasistatic_) {
    residual_ = mfem_ext::StdFunctionOperator(
        temperature_.space().TrueVSize(),

        [this](const mfem::Vector& u, mfem::Vector& r) {
          r = K_functional_(u);
          r.SetSubVector(bcs_.allEssentialDofs(), 0.0);
        },

        [this](const mfem::Vector& u) -> mfem::Operator& {
          auto& J = dynamic_cast<mfem::HypreParMatrix&>(K_functional_->GetGradient(u));
          bcs_.eliminateAllEssentialDofsFromMatrix(J);
          return J;
        });

  } else {
    // If dynamic, assemble the mass matrix
    M_functional_.AddDomainIntegral(
      Dimension<dim>{},
      [this]([[maybe_unused]] auto x, [[maybe_unused]] auto temperature) {
        auto source = cp_(x, t) * rho_(x, t);

        auto flux = zero{};

        // Return the source and the flux as a tuple 
        return serac::tuple{source, flux};
      },
      mesh_);

    residual_ = mfem_ext::StdFunctionOperator(
        temperature_.space().TrueVSize(),
        [this](const mfem::Vector& du_dt, mfem::Vector& r) {
          r = (M_functional.GetGradient(u_)) * du_dt + (*K_functional_) * (u_ + dt_ * du_dt);
          r.SetSubVector(bcs_.allEssentialDofs(), 0.0);
        },

        [this](const mfem::Vector& du_dt) -> mfem::Operator& {
          // Only reassemble the stiffness if it is a new timestep or we have a nonlinear reaction
          if (dt_ != previous_dt_) {
            auto localJ = std::unique_ptr<mfem::SparseMatrix>(
                mfem::Add(1.0, M_form_->SpMat(), dt_, K_form_->GetLocalGradient(u_ + dt_ * du_dt)));
            J_.reset(M_form_->ParallelAssemble(localJ.get()));
            bcs_.eliminateAllEssentialDofsFromMatrix(*J_);
          }
          return *J_;
        });
  }
}

template<int order, int dim>
void ThermalConductionFunctional<order, dim>::advanceTimestep(double& dt)
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

template<int order, int dim>
void ThermalConductionFunctional<order, dim>::InputOptions::defineInputFileSchema(axom::inlet::Container& container)
{
  // Polynomial interpolation order - currently up to 8th order is allowed
  container.addInt("order", "Order degree of the finite elements.").defaultValue(1).range(1, 8);

  // material parameters
  container.addDouble("kappa", "Thermal conductivity").defaultValue(0.5);
  container.addDouble("rho", "Density").defaultValue(1.0);
  container.addDouble("cp", "Specific heat capacity").defaultValue(1.0);

  auto& source = container.addStruct("source", "Scalar source term (RHS of the thermal conduction PDE)");
  serac::input::CoefficientInputOptions::defineInputFileSchema(source);

  auto& reaction_container = container.addStruct("nonlinear_reaction", "Nonlinear reaction term parameters");
  reaction_container.addFunction("reaction_function", axom::inlet::FunctionTag::Double,
                                 {axom::inlet::FunctionTag::Double}, "Nonlinear reaction function q = q(temperature)");
  reaction_container.addFunction("d_reaction_function", axom::inlet::FunctionTag::Double,
                                 {axom::inlet::FunctionTag::Double},
                                 "Derivative of the nonlinear reaction function dq = dq / dTemperature");
  auto& scale_coef_container = reaction_container.addStruct("scale", "Spatially varying scale factor for the reaction");
  serac::input::CoefficientInputOptions::defineInputFileSchema(scale_coef_container);

  auto& equation_solver_container =
      container.addStruct("equation_solver", "Linear and Nonlinear stiffness Solver Parameters.");
  serac::mfem_ext::EquationSolver::DefineInputFileSchema(equation_solver_container);

  auto& dynamics_container = container.addStruct("dynamics", "Parameters for mass matrix inversion");
  dynamics_container.addString("timestepper", "Timestepper (ODE) method to use");
  dynamics_container.addString("enforcement_method", "Time-varying constraint enforcement method to use");

  auto& bc_container = container.addStructDictionary("boundary_conds", "Container of boundary conditions");
  serac::input::BoundaryConditionInputOptions::defineInputFileSchema(bc_container);

  auto& init_temp = container.addStruct("initial_temperature", "Coefficient for initial condition");
  serac::input::CoefficientInputOptions::defineInputFileSchema(init_temp);
}

// force template instantiations
ThermalConductionFunctional<1, 2>;
ThermalConductionFunctional<2, 2>;
ThermalConductionFunctional<3, 2>;
ThermalConductionFunctional<4, 2>;

ThermalConductionFunctional<1, 3>;
ThermalConductionFunctional<2, 3>;
ThermalConductionFunctional<3, 3>;
ThermalConductionFunctional<4, 3>;

}  // namespace serac

using serac::DirichletEnforcementMethod;
using serac::ThermalConductionFunctional;
using serac::TimestepMethod;

serac::ThermalConductionFunctional::InputOptions FromInlet<serac::ThermalConductionFunctional::InputOptions>::operator()(
    const axom::inlet::Container& base)
{
  ThermalConductionFunctional::InputOptions result;

  result.order = base["order"];

  // Solver parameters
  auto equation_solver                   = base["equation_solver"];
  result.solver_options.T_lin_options    = equation_solver["linear"].get<serac::LinearSolverOptions>();
  result.solver_options.T_nonlin_options = equation_solver["nonlinear"].get<serac::NonlinearSolverOptions>();

  if (base.contains("dynamics")) {
    ThermalConductionFunctional::TimesteppingOptions dyn_options;
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

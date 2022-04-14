// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermal_conduction.hpp
 *
 * @brief An object containing the solver for a thermal conduction PDE
 */

#pragma once

#include "mfem.hpp"

#include "serac/physics/base_physics.hpp"
#include "serac/numerics/odes.hpp"
#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/functional_material_utils.hpp"
#include "serac/numerics/expr_template_ops.hpp"

namespace serac {

namespace Thermal {

/// A timestep and boundary condition enforcement method for a dynamic solver
struct TimesteppingOptions {
  /// The timestepping method to be applied
  serac::TimestepMethod timestepper;

  /// The essential boundary enforcement method to use
  serac::DirichletEnforcementMethod enforcement_method;
};

/**
 * @brief A configuration variant for the various solves
 * For quasistatic solves, leave the @a dyn_options parameter null. @a T_nonlin_options and @a T_lin_options
 * define the solver parameters for the nonlinear residual and linear stiffness solves. For
 * dynamic problems, @a dyn_options defines the timestepping scheme while @a T_lin_options and @a T_nonlin_options
 * define the nonlinear residual and linear stiffness solve options as before.
 */
struct SolverOptions {
  /// The linear solver options
  LinearSolverOptions T_lin_options;

  /// The nonlinear solver options
  NonlinearSolverOptions T_nonlin_options;

  /**
   * @brief The optional ODE solver parameters
   * @note If this is not defined, a quasi-static solve is performed
   */
  std::optional<TimesteppingOptions> dyn_options = std::nullopt;
};

/**
 * @brief Reasonable defaults for most thermal linear solver options
 *
 * @return The default thermal linear options
 */
IterativeSolverOptions defaultLinearOptions()
{
  return {.rel_tol     = 1.0e-6,
          .abs_tol     = 1.0e-12,
          .print_level = 0,
          .max_iter    = 200,
          .lin_solver  = LinearSolver::CG,
          .prec        = HypreSmootherPrec{mfem::HypreSmoother::Jacobi}};
}

/**
 * @brief Reasonable defaults for most thermal nonlinear solver options
 *
 * @return The default thermal nonlinear options
 */
NonlinearSolverOptions defaultNonlinearOptions()
{
  return {.rel_tol = 1.0e-4, .abs_tol = 1.0e-8, .max_iter = 500, .print_level = 1};
}

/**
 * @brief Reasonable defaults for quasi-static thermal conduction simulations
 *
 * @return The default quasi-static solver options
 */
SolverOptions defaultQuasistaticOptions() { return {defaultLinearOptions(), defaultNonlinearOptions(), std::nullopt}; }

/**
 * @brief Reasonable defaults for dynamic thermal conduction simulations
 *
 * @return The default dynamic solver options
 */
SolverOptions defaultDynamicOptions()
{
  return {defaultLinearOptions(), defaultNonlinearOptions(),
          Thermal::TimesteppingOptions{TimestepMethod::BackwardEuler, DirichletEnforcementMethod::RateControl}};
}

}  // namespace Thermal

/**
 * @brief An object containing the solver for a thermal conduction PDE
 *
 * This is a generic linear thermal diffusion oeprator of the form
 *
 * \f[
 * \mathbf{M} \frac{\partial \mathbf{u}}{\partial t} = -\kappa \mathbf{K} \mathbf{u} + \mathbf{f}
 * \f]
 *
 *  where \f$\mathbf{M}\f$ is a mass matrix, \f$\mathbf{K}\f$ is a stiffness matrix, \f$\mathbf{u}\f$ is the
 *  temperature degree of freedom vector, and \f$\mathbf{f}\f$ is a thermal load vector.
 */

template <int order, int dim, typename... parameter_space>
class ThermalConductionFunctional : public BasePhysics {
public:
  /**
   * @brief Construct a new Thermal Functional Solver object
   *
   * @param[in] options The system linear and nonlinear solver and timestepping parameters
   * @param[in] name An optional name for the physics module instance
   * @param[in] parameter_states The optional array of finite element states represetnting user-defined parameters to be
   * used by an underlying material model or load
   */
  ThermalConductionFunctional(
      const Thermal::SolverOptions& options, const std::string& name = {},
      std::array<std::reference_wrapper<FiniteElementState>, sizeof...(parameter_space)> parameter_states = {})
      : BasePhysics(2, order),
        temperature_(
            StateManager::newState(FiniteElementState::Options{.order      = order,
                                                               .vector_dim = 1,
                                                               .ordering   = mfem::Ordering::byNODES,
                                                               .name       = detail::addPrefix(name, "temperature")})),
        adjoint_temperature_(StateManager::newState(
            FiniteElementState::Options{.order      = order,
                                        .vector_dim = 1,
                                        .ordering   = mfem::Ordering::byNODES,
                                        .name       = detail::addPrefix(name, "adjoint_temperature")})),
        parameter_states_(parameter_states),
        residual_(temperature_.space().TrueVSize()),
        ode_(temperature_.space().TrueVSize(), {.u = u_, .dt = dt_, .du_dt = previous_, .previous_dt = previous_dt_},
             nonlin_solver_, bcs_)
  {
    SLIC_ERROR_ROOT_IF(mesh_.Dimension() != dim,
                       axom::fmt::format("Compile time dimension and runtime mesh dimension mismatch"));

    // Create a pack of the primal field and parameter finite element spaces
    std::array<mfem::ParFiniteElementSpace*, sizeof...(parameter_space) + 1> trial_spaces;
    trial_spaces[0] = &temperature_.space();

    functional_call_args_.emplace_back(temperature_.trueVec());

    if constexpr (sizeof...(parameter_space) > 0) {
      for (size_t i = 0; i < sizeof...(parameter_space); ++i) {
        trial_spaces[i + 1]         = &(parameter_states_[i].get().space());
        parameter_sensitivities_[i] = std::make_unique<FiniteElementDual>(mesh_, parameter_states_[i].get().space());
        functional_call_args_.emplace_back(parameter_states_[i].get().trueVec());
      }
    }

    M_functional_ = std::make_unique<Functional<test(trial, parameter_space...)>>(&temperature_.space(), trial_spaces);

    K_functional_ = std::make_unique<Functional<test(trial, parameter_space...)>>(&temperature_.space(), trial_spaces);

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
  }

  /**
   * @brief Set essential temperature boundary conditions (strongly enforced)
   *
   * @param[in] temp_bdr The boundary attributes on which to enforce a temperature
   * @param[in] temp The prescribed boundary temperature function
   */
  void setTemperatureBCs(const std::set<int>& temp_bdr, std::function<double(const mfem::Vector& x, double t)> temp)
  {
    // Project the coefficient onto the grid function
    temp_bdr_coef_ = std::make_shared<mfem::FunctionCoefficient>(temp);

    bcs_.addEssential(temp_bdr, temp_bdr_coef_, temperature_);
  }

  /**
   * @brief Advance the timestep
   *
   * @param[inout] dt The timestep to advance. For adaptive time integration methods, the actual timestep is returned.
   */
  void advanceTimestep(double& dt) override
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

  /**
   * @brief Set the thermal flux and mass properties for the physics module
   *
   * @tparam MaterialType The thermal material type
   * @param material A material containing density, specific heat, and thermal flux evaluation information
   *
   * @pre MaterialType must have a method specificHeatCapacity() defining the specific heat
   * @pre MaterialType must have a method density() defining the density
   * @pre MaterialType must have the operator (temperature, d temperature_dx) defined as the thermal flux
   */
  template <typename MaterialType>
  void setMaterial(MaterialType material)
  {
    if constexpr (is_parameterized<MaterialType>::value) {
      static_assert(material.numParameters() == sizeof...(parameter_space),
                    "Number of parameters in thermal conduction does not equal the number of parameters in the "
                    "thermal material.");
    }

    auto parameterized_material = parameterizeMaterial(material);

    K_functional_->AddDomainIntegral(
        Dimension<dim>{},
        [parameterized_material](auto x, auto temperature, auto... params) {
          // Get the value and the gradient from the input tuple
          auto [u, du_dx] = temperature;
          auto source     = serac::zero{};

          auto response = parameterized_material(x, u, du_dx, serac::get<0>(params)...);

          return serac::tuple{source, -1.0 * response.heat_flux};
        },
        mesh_);

    M_functional_->AddDomainIntegral(
        Dimension<dim>{},
        [parameterized_material](auto x, auto d_temperature_dt, auto... params) {
          auto [u, du_dx] = d_temperature_dt;
          auto flux       = serac::zero{};

          auto temp      = u * 0.0;
          auto temp_grad = du_dx * 0.0;

          auto response = parameterized_material(x, temp, temp_grad, serac::get<0>(params)...);

          auto source = response.specific_heat_capacity * response.density * u;

          // Return the source and the flux as a tuple
          return serac::tuple{source, flux};
        },
        mesh_);
  }

  /**
   * @brief Set the underlying finite element state to a prescribed temperature
   *
   * @param temp The function describing the temperature field
   */
  void setTemperature(std::function<double(const mfem::Vector& x, double t)> temp)
  {
    // Project the coefficient onto the grid function
    mfem::FunctionCoefficient temp_coef(temp);

    temp_coef.SetTime(time_);
    temperature_.project(temp_coef);
    gf_initialized_[0] = true;
  }

  /**
   * @brief Set the thermal source function
   *
   * @tparam SourceType The type of the source function
   * @param source_function A source function for a prescribed thermal load
   *
   * @pre SourceType must have the operator (x, time, temperature, d temperature_dx) defined as the thermal source
   */
  template <typename SourceType>
  void setSource(SourceType source_function)
  {
    if constexpr (is_parameterized<SourceType>::value) {
      static_assert(source_function.numParameters() == sizeof...(parameter_space),
                    "Number of parameters in thermal conduction does not equal the number of parameters in the "
                    "thermal source.");
    }

    auto parameterized_source = parameterizeSource(source_function);

    K_functional_->AddDomainIntegral(
        Dimension<dim>{},
        [parameterized_source, this](auto x, auto temperature, auto... params) {
          // Get the value and the gradient from the input tuple
          auto [u, du_dx] = temperature;

          auto flux = serac::zero{};

          auto source = -1.0 * parameterized_source(x, time_, u, du_dx, serac::get<0>(params)...);

          // Return the source and the flux as a tuple
          return serac::tuple{source, flux};
        },
        mesh_);
  }

  /**
   * @brief Set the thermal flux boundary condition
   *
   * @tparam FluxType The type of the flux function
   * @param flux_function A function describing the thermal flux applied to a boundary
   *
   * @pre FluxType must have the operator (x, normal, temperature) to return the thermal flux value
   */
  template <typename FluxType>
  void setFluxBCs(FluxType flux_function)
  {
    if constexpr (is_parameterized<FluxType>::value) {
      static_assert(flux_function.numParameters() == sizeof...(parameter_space),
                    "Number of parameters in thermal conduction does not equal the number of parameters in the "
                    "thermal flux boundary.");
    }

    auto parameterized_flux = parameterizeFlux(flux_function);

    K_functional_->AddBoundaryIntegral(
        Dimension<dim - 1>{},
        [parameterized_flux](auto x, auto n, auto u, auto... params) { return parameterized_flux(x, n, u, params...); },
        mesh_);
  }

  /**
   * @brief Get the temperature state
   *
   * @return A reference to the current temperature finite element state
   */
  const serac::FiniteElementState& temperature() const { return temperature_; };

  /// @overload
  serac::FiniteElementState& temperature() { return temperature_; };

  /**
   * @brief Get the adjoint temperature state
   *
   * @return A reference to the current adjoint temperature finite element state
   */
  const serac::FiniteElementState& adjointTemperature() const { return adjoint_temperature_; };

  /// @overload
  serac::FiniteElementState& adjointTemperature() { return adjoint_temperature_; };

  /**
   * @brief Complete the initialization and allocation of the data structures.
   *
   * This must be called before AdvanceTimestep().
   */
  void completeSetup() override
  {
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
            functional_call_args_[0] = u;

            r = (*K_functional_)(functional_call_args_);
            r.SetSubVector(bcs_.allEssentialDofs(), 0.0);
          },

          [this](const mfem::Vector& u) -> mfem::Operator& {
            functional_call_args_[0] = u;

            auto [r, drdu] = (*K_functional_)(functional_call_args_, Index<0>{});
            J_             = assemble(drdu);
            bcs_.eliminateAllEssentialDofsFromMatrix(*J_);
            return *J_;
          });

    } else {
      // If dynamic, assemble the mass matrix
      residual_ = mfem_ext::StdFunctionOperator(
          temperature_.space().TrueVSize(),
          [this](const mfem::Vector& du_dt, mfem::Vector& r) {
            functional_call_args_[0] = du_dt;

            auto M_residual = (*M_functional_)(functional_call_args_);

            // TODO we should use the new variadic capability to directly pass temperature and d_temp_dt directly to
            // these kernels to avoid ugly hacks like this.
            mfem::Vector K_arg(u_.Size());
            add(1.0, u_, dt_, du_dt, K_arg);
            functional_call_args_[0] = K_arg;

            auto K_residual = (*K_functional_)(functional_call_args_);

            functional_call_args_[0] = u_;

            add(M_residual, K_residual, r);
            r.SetSubVector(bcs_.allEssentialDofs(), 0.0);
          },

          [this](const mfem::Vector& du_dt) -> mfem::Operator& {
            // Only reassemble the stiffness if it is a new timestep
            functional_call_args_[0] = du_dt;

            auto M = serac::get<1>((*M_functional_)(functional_call_args_, Index<0>{}));
            std::unique_ptr<mfem::HypreParMatrix> m_mat(assemble(M));

            mfem::Vector K_arg(u_.Size());
            add(1.0, u_, dt_, du_dt, K_arg);
            functional_call_args_[0] = K_arg;

            auto K = serac::get<1>((*K_functional_)(functional_call_args_, Index<0>{}));

            functional_call_args_[0] = u_;

            std::unique_ptr<mfem::HypreParMatrix> k_mat(assemble(K));

            J_.reset(mfem::Add(1.0, *m_mat, dt_, *k_mat));
            bcs_.eliminateAllEssentialDofsFromMatrix(*J_);
            return *J_;
          });
    }
  }

  /**
   * @brief Solve the adjoint problem
   * @pre It is expected that the forward analysis is complete and the current temperature state is valid
   * @note If the essential boundary state is not specified, homogeneous essential boundary conditions are applied
   *
   * @param[in] adjoint_load The dual state that contains the right hand side of the adjoint system (d quantity of
   * interest/d temperature)
   * @param[in] dual_with_essential_boundary A optional finite element dual containing the non-homogenous essential
   * boundary condition data for the adjoint problem
   * @return The computed adjoint finite element state
   */
  virtual const serac::FiniteElementState& solveAdjoint(FiniteElementDual& adjoint_load,
                                                        FiniteElementDual* dual_with_essential_boundary = nullptr)
  {
    mfem::HypreParVector adjoint_load_vector(adjoint_load.trueVec());

    // Add the sign correction to move the term to the RHS
    adjoint_load_vector *= -1.0;

    auto& lin_solver = nonlin_solver_.LinearSolver();

    // By default, use a homogeneous essential boundary condition
    mfem::HypreParVector adjoint_essential(adjoint_load.trueVec());
    adjoint_essential = 0.0;

    functional_call_args_[0] = temperature_.trueVec();

    auto [r, drdu] = (*K_functional_)(functional_call_args_, Index<0>{});
    auto jacobian  = assemble(drdu);
    auto J_T       = std::unique_ptr<mfem::HypreParMatrix>(jacobian->Transpose());

    // If we have a non-homogeneous essential boundary condition, extract it from the given state
    if (dual_with_essential_boundary) {
      dual_with_essential_boundary->initializeTrueVec();
      adjoint_essential = dual_with_essential_boundary->trueVec();
    }

    for (const auto& bc : bcs_.essentials()) {
      bc.eliminateFromMatrix(*J_T);
      bc.eliminateToRHS(*J_T, adjoint_essential, adjoint_load_vector);
    }

    lin_solver.SetOperator(*J_T);
    lin_solver.Mult(adjoint_load_vector, adjoint_temperature_.trueVec());

    adjoint_temperature_.distributeSharedDofs();

    // Reset the equation solver to use the full nonlinear residual operator
    nonlin_solver_.SetOperator(residual_);

    return adjoint_temperature_;
  }

  /**
   * @brief Compute the implicit sensitivity of the quantity of interest used in defining the load for the adjoint
   * problem with respect to the parameter field
   *
   * @tparam parameter_field The index of the parameter to take a derivative with respect to
   * @return The sensitivity with respect to the parameter
   *
   * @pre `solveAdjoint` with an appropriate adjoint load must be called prior to this method.
   */
  template <int parameter_field>
  FiniteElementDual& computeSensitivity()
  {
    functional_call_args_[0] = temperature_.trueVec();

    auto [r, drdparam] = (*K_functional_)(functional_call_args_, Index<parameter_field + 1>{});

    auto drdparam_mat = assemble(drdparam);

    drdparam_mat->MultTranspose(adjoint_temperature_.trueVec(), parameter_sensitivities_[parameter_field]->trueVec());

    parameter_sensitivities_[parameter_field]->distributeSharedDofs();

    return *parameter_sensitivities_[parameter_field];
  }

  /// Destroy the Thermal Solver object
  virtual ~ThermalConductionFunctional() = default;

protected:
  /// The compile-time finite element trial space for thermal conduction (H1 of order p)
  using trial = H1<order>;

  /// The compile-time finite element test space for thermal conduction (H1 of order p)
  using test = H1<order>;

  /// The temperature finite element state
  serac::FiniteElementState temperature_;

  /// The adjoint temperature finite element state
  serac::FiniteElementState adjoint_temperature_;

  /// Mass functional object \f$\mathbf{M} = \int_\Omega c_p \, \rho \, \phi_i \phi_j\, dx \f$
  std::unique_ptr<Functional<test(trial, parameter_space...)>> M_functional_;

  /// Stiffness functional object \f$\mathbf{K} = \int_\Omega \theta \cdot \nabla \phi_i  + f \phi_i \, dx \f$
  std::unique_ptr<Functional<test(trial, parameter_space...)>> K_functional_;

  /// The finite element states representing user-defined parameter fields
  std::array<std::reference_wrapper<FiniteElementState>, sizeof...(parameter_space)> parameter_states_;

  /// The sensitivities (dual vectors) with repect to each of the input parameter fields
  std::array<std::unique_ptr<FiniteElementDual>, sizeof...(parameter_space)> parameter_sensitivities_;

  /// The set of input trial space vectors (temperature + parameters) used to call the underlying functional
  std::vector<std::reference_wrapper<const mfem::Vector>> functional_call_args_;

  /// Assembled mass matrix
  std::unique_ptr<mfem::HypreParMatrix> M_;

  /// Coefficient containing the essential boundary values
  std::shared_ptr<mfem::Coefficient> temp_bdr_coef_;

  /**
   * @brief mfem::Operator that describes the weight residual
   * and its gradient with respect to temperature
   */
  mfem_ext::StdFunctionOperator residual_;

  /**
   * @brief the ordinary differential equation that describes
   * how to solve for the time derivative of temperature, given
   * the current temperature and source terms
   */
  mfem_ext::FirstOrderODE ode_;

  /// the specific methods and tolerances specified to solve the nonlinear residual equations
  mfem_ext::EquationSolver nonlin_solver_;

  /// Assembled sparse matrix for the Jacobian
  std::unique_ptr<mfem::HypreParMatrix> J_;

  /// The current timestep
  double dt_;

  /// The previous timestep
  double previous_dt_;

  /// An auxilliary zero vector
  mfem::Vector zero_;

  /// Predicted temperature true dofs
  mfem::Vector u_;

  /// Previous value of du_dt used to prime the pump for the nonlinear solver
  mfem::Vector previous_;
};

}  // namespace serac

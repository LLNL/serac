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
#include "serac/physics/materials/thermal_functional_material.hpp"

namespace serac {

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
template <int order, int dim>
class ThermalConductionFunctional : public BasePhysics {
public:
  /// A timestep and boundary condition enforcement method for a dynamic solver
  struct TimesteppingOptions {
    /// The timestepping method to be applied
    TimestepMethod timestepper;

    /// The essential boundary enforcement method to use
    DirichletEnforcementMethod enforcement_method;
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
  static IterativeSolverOptions defaultLinearOptions()
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
  static NonlinearSolverOptions defaultNonlinearOptions()
  {
    return {.rel_tol = 1.0e-4, .abs_tol = 1.0e-8, .max_iter = 500, .print_level = 1};
  }

  /**
   * @brief Reasonable defaults for quasi-static thermal conduction simulations
   *
   * @return The default quasi-static solver options
   */
  static SolverOptions defaultQuasistaticOptions()
  {
    return {defaultLinearOptions(), defaultNonlinearOptions(), std::nullopt};
  }

  /**
   * @brief Reasonable defaults for dynamic thermal conduction simulations
   *
   * @return The default dynamic solver options
   */
  static SolverOptions defaultDynamicOptions()
  {
    return {defaultLinearOptions(), defaultNonlinearOptions(),
            TimesteppingOptions{TimestepMethod::BackwardEuler, DirichletEnforcementMethod::RateControl}};
  }

  /**
   * @brief Construct a new Thermal Functional Solver object
   *
   * @param[in] options The system linear and nonlinear solver and timestepping parameters
   * @param[in] name An optional name for the physics module instance
   */
  ThermalConductionFunctional(const SolverOptions& options, const std::string& name = "")
      : BasePhysics(1, order),
        temperature_(
            StateManager::newState(FiniteElementState::Options{.order      = order,
                                                               .vector_dim = 1,
                                                               .ordering   = mfem::Ordering::byNODES,
                                                               .name       = detail::addPrefix(name, "temperature")})),
        M_functional_(&temperature_.space(), {&temperature_.space()}),
        K_functional_(&temperature_.space(), {&temperature_.space()}),
        residual_(temperature_.space().TrueVSize()),
        ode_(temperature_.space().TrueVSize(), {.u = u_, .dt = dt_, .du_dt = previous_, .previous_dt = previous_dt_},
             nonlin_solver_, bcs_)
  {
    SLIC_ERROR_ROOT_IF(mesh_.Dimension() != dim,
                       axom::fmt::format("Compile time dimension and runtime mesh dimension mismatch"));

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
    static_assert(Thermal::has_density<MaterialType, dim>::value,
                  "Thermal functional materials must have a public density(x) method.");
    static_assert(Thermal::has_specific_heat_capacity<MaterialType, dim>::value,
                  "Thermal functional materials must have a public specificHeatCapacity(x, temperature) method.");
    static_assert(Thermal::has_thermal_flux<MaterialType, dim>::value,
                  "Thermal functional materials must have a public (u, du_dx) operator for thermal flux evaluation.");

    K_functional_.AddDomainIntegral(
        Dimension<dim>{},
        [material](auto, auto temperature) {
          // Get the value and the gradient from the input tuple
          auto [u, du_dx] = temperature;
          auto flux       = -1.0 * material(u, du_dx);

          auto source = u * 0.0;

          // Return the source and the flux as a tuple
          return serac::tuple{source, flux};
        },
        mesh_);

    M_functional_.AddDomainIntegral(
        Dimension<dim>{},
        [material](auto x, auto temperature) {
          auto [u, du_dx] = temperature;

          auto source = material.specificHeatCapacity(x, u) * material.density(x);

          auto flux = 0.0 * du_dx;

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
    static_assert(
        Thermal::has_thermal_source<SourceType, dim>::value,
        "Thermal functional sources must have a public (x, t, u, du_dx) operator for thermal source evaluation.");

    K_functional_.AddDomainIntegral(
        Dimension<dim>{},
        [source_function, this](auto x, auto temperature) {
          // Get the value and the gradient from the input tuple
          auto [u, du_dx] = temperature;

          auto flux = du_dx * 0.0;

          auto source = -1.0 * source_function(x, time_, u, du_dx);

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
    static_assert(Thermal::has_thermal_flux_boundary<FluxType, dim>::value,
                  "Thermal flux boundary condition types must have a public (x, n, u) operator for thermal boundary "
                  "flux evaluation.");

    K_functional_.AddBoundaryIntegral(
        Dimension<dim - 1>{}, [flux_function](auto x, auto n, auto u) { return flux_function(x, n, u); }, mesh_);
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
      K_functional_.SetEssentialBC(bc.markers(), 0);
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
            auto [r, drdu] = K_functional_(differentiate_wrt(u));
            J_             = assemble(drdu);
            bcs_.eliminateAllEssentialDofsFromMatrix(*J_);
            return *J_;
          });

    } else {
      // If dynamic, assemble the mass matrix
      residual_ = mfem_ext::StdFunctionOperator(
          temperature_.space().TrueVSize(),
          [this](const mfem::Vector& du_dt, mfem::Vector& r) {
            mfem::Vector K_arg(u_.Size());
            add(1.0, u_, dt_, du_dt, K_arg);

            add(M_functional_(du_dt), K_functional_(K_arg), r);
            r.SetSubVector(bcs_.allEssentialDofs(), 0.0);
          },

          [this](const mfem::Vector& du_dt) -> mfem::Operator& {
            // Only reassemble the stiffness if it is a new timestep
            if (dt_ != previous_dt_) {
              mfem::Vector K_arg(u_.Size());
              add(1.0, u_, dt_, du_dt, K_arg);

              auto                                  M = serac::get<1>(M_functional_(differentiate_wrt(u_)));
              std::unique_ptr<mfem::HypreParMatrix> m_mat(assemble(M));

              auto                                  K = serac::get<1>(K_functional_(differentiate_wrt(K_arg)));
              std::unique_ptr<mfem::HypreParMatrix> k_mat(assemble(K));

              J_.reset(mfem::Add(1.0, *m_mat, dt_, *k_mat));
              bcs_.eliminateAllEssentialDofsFromMatrix(*J_);
            }
            return *J_;
          });
    }
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

  /// Mass functional object \f$\mathbf{M} = \int_\Omega c_p \, \rho \, \phi_i \phi_j\, dx \f$
  Functional<test(trial)> M_functional_;

  /// Stiffness functional object \f$\mathbf{K} = \int_\Omega \theta \cdot \nabla \phi_i  + f \phi_i \, dx \f$
  Functional<test(trial)> K_functional_;

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

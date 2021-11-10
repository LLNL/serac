// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
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

namespace serac {

namespace Thermal {

struct LinearIsotropicConductor {
  double rho;
  double cp;
  double kappa;

  template <typename T1, typename T2>
  SERAC_HOST_DEVICE T2 operator()([[maybe_unused]] T1& u, [[maybe_unused]] T2& du_dx) const
  {
    return kappa * du_dx;
  }
};

struct ConstantSource {
  double source;

  template <typename T1, typename T2>
  SERAC_HOST_DEVICE T2 operator()([[maybe_unused]] T1& u, [[maybe_unused]] T2& du_dx) const
  {
    return source;
  }
};

}  // namespace Thermal

/**
 * @brief An object containing the solver for a thermal conduction PDE
 *
 * This is a generic linear thermal diffusion oeprator of the form
 *
 *    M du/dt = -kappa Ku + f
 *
 *  where M is a mass matrix, K is a stiffness matrix, and f is a
 *  thermal load vector.
 */
template <int order, int dim>
class ThermalConductionFunctional : public BasePhysics {
public:
  /**
   * @brief A timestep method and config for the M solver
   */
  struct TimesteppingOptions {
    /**
     * @brief The timestepping method to be applied
     *
     */
    TimestepMethod timestepper;

    /**
     * @brief The essential boundary enforcement method to use
     *
     */
    DirichletEnforcementMethod enforcement_method;
  };

  /**
   * @brief A configuration variant for the various solves
   * Either quasistatic, or time-dependent with timestep and M options
   */
  struct SolverOptions {
    /**
     * @brief The linear solver options
     *
     */
    LinearSolverOptions T_lin_options;

    /**
     * @brief The nonlinear solver options
     *
     */
    NonlinearSolverOptions T_nonlin_options;

    /**
     * @brief The optional ODE solver parameters
     * @note If this is not defined, a quasi-static solve is performed
     *
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
   * @brief Construct a new Thermal Solver object
   *
   * @param[in] options The system solver parameters
   * @param[in] name An optional name for the physics module instance
   */
  ThermalConductionFunctional(const SolverOptions& options, const std::string& name = "")
      : BasePhysics(1, order),
        temperature_(
            StateManager::newState(FiniteElementState::Options{.order      = order,
                                                               .vector_dim = 1,
                                                               .ordering   = mfem::Ordering::byNODES,
                                                               .name       = detail::addPrefix(name, "temperature")})),
        M_functional_(&temperature_.space(), &temperature_.space()),
        K_functional_(&temperature_.space(), &temperature_.space()),
        residual_(temperature_.space().TrueVSize()),
        ode_(temperature_.space().TrueVSize(), {.u = u_, .dt = dt_, .du_dt = previous_, .previous_dt = previous_dt_},
             nonlin_solver_, bcs_)
  {
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
  }

  /**
   * @brief Set essential temperature boundary conditions (strongly enforced)
   *
   * @param[in] temp_bdr The boundary attributes on which to enforce a temperature
   * @param[in] temp_bdr_coef The prescribed boundary temperature
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

  template <typename MaterialType>
  void setMaterial(MaterialType material)
  {
    K_functional_.AddDomainIntegral(
        Dimension<dim>{},
        [material]([[maybe_unused]] auto x, auto temperature) {
          // Get the value and the gradient from the input tuple
          auto [u, du_dx] = temperature;

          auto flux = material(u, du_dx);

          auto source = u * 0.0;

          // Return the source and the flux as a tuple
          return serac::tuple{source, flux};
        },
        mesh_);

    M_functional_.AddDomainIntegral(
        Dimension<dim>{},
        [material]([[maybe_unused]] auto x, [[maybe_unused]] auto temperature) {
          auto [u, du_dx] = temperature;

          auto source = material.cp * material.rho;

          auto flux = 0.0 * du_dx;

          // Return the source and the flux as a tuple
          return serac::tuple{source, flux};
        },
        mesh_);
  }

  void setTemperature(std::function<double(const mfem::Vector& x, double t)> temp)
  {
    // Project the coefficient onto the grid function
    mfem::FunctionCoefficient temp_coef(temp);

    temp_coef.SetTime(time_);
    temperature_.project(temp_coef);
    gf_initialized_[0] = true;
  }

  template <typename SourceType>
  void setSource(SourceType source_function)
  {
    K_functional_.AddDomainIntegral(
        Dimension<dim>{},
        [source_function, this]([[maybe_unused]] auto x, auto temperature) {
          // Get the value and the gradient from the input tuple
          auto [u, du_dx] = temperature;

          auto flux = du_dx * 0.0;

          auto source = source_function(x, time_);

          // Return the source and the flux as a tuple
          return serac::tuple{source, flux};
        },
        mesh_);
  }

  /**
   * @brief Get the temperature state
   *
   * @return A reference to the current temperature finite element state
   */
  const serac::FiniteElementState& temperature() const { return temperature_; };

  /**
   * @overload
   */
  serac::FiniteElementState& temperature() { return temperature_; };

  /**
   * @brief Complete the initialization and allocation of the data structures.
   *
   * This must be called before StaticSolve() or AdvanceTimestep(). If allow_dynamic
   * = false, do not allocate the mass matrix or dynamic operator
   */
  void completeSetup() override
  {
    // Build the dof array lookup tables
    temperature_.space().BuildDofToArrays();

    // Project the essential boundary coefficients
    for (auto& bc : bcs_.essentials()) {
      bc.projectBdr(temperature_, time_);
      K_functional_.SetEssentialBC(bc.markers());
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
            K_functional_(u);
            J_.reset(grad(K_functional_));
            bcs_.eliminateAllEssentialDofsFromMatrix(*J_);
            return *J_;
          });

    } else {
      // If dynamic, assemble the mass matrix
      residual_ = mfem_ext::StdFunctionOperator(
          temperature_.space().TrueVSize(),
          [this](const mfem::Vector& du_dt, mfem::Vector& r) {
            mfem::Vector K_arg;
            add(1.0, u_, dt_, du_dt, K_arg);

            add(M_functional_(du_dt), K_functional_(K_arg), r);
            r.SetSubVector(bcs_.allEssentialDofs(), 0.0);
          },

          [this](const mfem::Vector& du_dt) -> mfem::Operator& {
            // Only reassemble the stiffness if it is a new timestep
            if (dt_ != previous_dt_) {
              mfem::Vector K_arg;
              add(1.0, u_, dt_, du_dt, K_arg);

              M_functional_(u_);
              std::unique_ptr<mfem::HypreParMatrix> m_mat(grad(M_functional_));

              K_functional_(K_arg);
              std::unique_ptr<mfem::HypreParMatrix> k_mat(grad(K_functional_));

              J_.reset(mfem::Add(1.0, *m_mat, dt_, *k_mat));
              bcs_.eliminateAllEssentialDofsFromMatrix(*J_);
            }
            return *J_;
          });
    }
  }

  /**
   * @brief Destroy the Thermal Solver object
   */
  virtual ~ThermalConductionFunctional() = default;

protected:
  using trial = H1<order>;
  using test  = H1<order>;

  /**
   * @brief The temperature finite element state
   */
  serac::FiniteElementState temperature_;

  /**
   * @brief Mass bilinear form object
   */
  Functional<test(trial)> M_functional_;

  /**
   * @brief Stiffness nonlinear form object
   */
  Functional<test(trial)> K_functional_;

  /**
   * @brief Assembled mass matrix
   */
  std::unique_ptr<mfem::HypreParMatrix> M_;

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

  /**
   * @brief the specific methods and tolerances specified to
   * solve the nonlinear residual equations
   */
  mfem_ext::EquationSolver nonlin_solver_;

  /**
   * @brief assembled sparse matrix for the Jacobian
   * at the predicted temperature
   */
  std::unique_ptr<mfem::HypreParMatrix> J_;

  /**
   * @brief The current timestep
   */
  double dt_;

  /**
   * @brief The previous timestep
   */
  double previous_dt_;

  /**
   * @brief A zero vector
   */
  mfem::Vector zero_;

  /**
   * @brief predicted temperature true dofs
   */
  mfem::Vector u_;

  /**
   * @brief previous value of du_dt used to prime the pump for the
   * nonlinear solver
   */
  mfem::Vector previous_;
};

}  // namespace serac

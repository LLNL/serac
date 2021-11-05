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

namespace serac {

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
  using ScalarFunction = std::function<tensor<double, 1>(tensor<double, dim>, tensor<double, 1>)>;
  using VectorFunction = std::function<tensor<double, dim>(tensor<double, dim>, tensor<double, 1>)>;

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
  ThermalConductionFunctional(const SolverOptions& options, const std::string& name = "");

  /**
   * @brief Set essential temperature boundary conditions (strongly enforced)
   *
   * @param[in] temp_bdr The boundary attributes on which to enforce a temperature
   * @param[in] temp_bdr_coef The prescribed boundary temperature
   */
  void setTemperatureBCs(const std::set<int>& temp_bdr, ScalarFunction temp_function);

  /**
   * @brief Advance the timestep
   *
   * @param[inout] dt The timestep to advance. For adaptive time integration methods, the actual timestep is returned.
   */
  void advanceTimestep(double& dt) override;

  /**
   * @brief Set the thermal conductivity
   *
   * @param[in] kappa The thermal conductivity
   */
  void setConductivity(ScalarFunction kappa_function);

  /**
   * @brief Set the temperature state vector from a coefficient
   *
   * @param[in] temp The temperature coefficient
   */
  void setTemperature(ScalarFunction temp_function);

  /**
   * @brief Set the thermal body source from a coefficient
   *
   * @param[in] source The source function coefficient
   */
  void setSource(ScalarFunction source_function);

  /**
   * @brief Set the density field. Defaults to 1.0 if not set.
   *
   * @param[in] rho The density field coefficient
   */
  void setMassDensity(ScalarFunction rho_function);

  /**
   * @brief Set the specific heat capacity. Defaults to 1.0 if not set.
   *
   * @param[in] cp The specific heat capacity
   */
  void setSpecificHeatCapacity(ScalarFunction cp_function);

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
  void completeSetup() override;

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
   * @brief Conduction coefficient
   */
  ScalarFunction kappa_;

  /**
   * @brief Body source coefficient
   */
  ScalarFunction source_;

  /**
   * @brief Density coefficient
   *
   */
  ScalarFunction rho_;

  /**
   * @brief Specific heat capacity
   *
   */
  ScalarFunction cp_;

  /**
   * @brief Combined mass matrix coefficient (rho * cp)
   *
   */
  ScalarFunction mass_coef_;

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

// force template instantiations
template class ThermalConductionFunctional<1, 2>;

}  // namespace serac

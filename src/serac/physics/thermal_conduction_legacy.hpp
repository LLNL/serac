// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermal_conduction_legacy.hpp
 *
 * @brief An object containing the solver for a thermal conduction PDE
 */

#pragma once

#include "mfem.hpp"

#include "serac/physics/base_physics.hpp"
#include "serac/numerics/odes.hpp"
#include "serac/numerics/stdfunction_operator.hpp"

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
class ThermalConductionLegacy : public BasePhysics {
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
   * @brief Stores all information held in the input file that
   * is used to configure the solver
   */
  struct InputOptions {
    /**
     * @brief Input file parameters specific to this class
     *
     * @param[in] container Inlet's Container that input files will be added to
     **/
    static void defineInputFileSchema(axom::inlet::Container& container);

    /**
     * @brief The order of the discretized field
     *
     */
    int order;

    /**
     * @brief The linear, nonlinear, and ODE solver options
     *
     */
    SolverOptions solver_options;

    /**
     * @brief The conductivity parameter
     *
     */
    double kappa;

    /**
     * @brief The specific heat capacity
     *
     */
    double cp;

    /**
     * @brief The mass density
     *
     */
    double rho;

    /**
     * @brief Reaction function r(T)
     *
     */
    std::function<double(double)> reaction_func;

    /**
     * @brief Derivative of the reaction function dR(T)/dT
     *
     */
    std::function<double(double)> d_reaction_func;

    /**
     * @brief The coefficient options for the scaling factor
     *
     */
    std::optional<input::CoefficientInputOptions> reaction_scale_coef;

    /**
     * @brief Source function coefficient
     *
     */
    std::optional<input::CoefficientInputOptions> source_coef;

    /**
     * @brief The boundary condition information
     */
    std::unordered_map<std::string, input::BoundaryConditionInputOptions> boundary_conditions;

    /**
     * @brief The initial temperature field
     * @note This can be used as either an intialization for dynamic simulations or an
     *       initial guess for quasi-static ones
     *
     */
    std::optional<input::CoefficientInputOptions> initial_temperature;
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
   * @param[in] order The order of the thermal field discretization
   * @param[in] options The system solver parameters
   * @param[in] name An optional name for the physics module instance
   * @param[in] pmesh An optional mesh reference, must be provided to configure the module
   * when a mesh other than the default mesh is used
   */
  ThermalConductionLegacy(int order, const SolverOptions& options, const std::string& name = "",
                          mfem::ParMesh* pmesh = nullptr);

  /**
   * @brief Construct a new Thermal Solver object
   *
   * @param[in] options The solver information parsed from the input file
   * @param[in] name An optional name for the physics module instance
   */
  ThermalConductionLegacy(const InputOptions& options, const std::string& name = "");

  /**
   * @brief Set essential temperature boundary conditions (strongly enforced)
   *
   * @param[in] temp_bdr The boundary attributes on which to enforce a temperature
   * @param[in] temp_bdr_coef The prescribed boundary temperature
   */
  void setTemperatureBCs(const std::set<int>& temp_bdr, std::shared_ptr<mfem::Coefficient> temp_bdr_coef);

  /**
   * @brief Set flux boundary conditions (weakly enforced)
   *
   * @param[in] flux_bdr The boundary attributes on which to enforce a heat flux (weakly enforced)
   * @param[in] flux_bdr_coef The prescribed boundary heat flux
   */
  void setFluxBCs(const std::set<int>& flux_bdr, std::shared_ptr<mfem::Coefficient> flux_bdr_coef);

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
  void setConductivity(std::unique_ptr<mfem::Coefficient>&& kappa);

  /**
   * @brief Set the temperature state vector from a coefficient
   *
   * @param[in] temp The temperature coefficient
   */
  void setTemperature(mfem::Coefficient& temp);

  /**
   * @brief Set the thermal body source from a coefficient
   *
   * @param[in] source The source function coefficient
   */
  void setSource(std::unique_ptr<mfem::Coefficient>&& source);

  /**
   * @brief Set a nonlinear temperature dependent reaction term
   *
   * @param[in] reaction A function describing the temperature dependent reaction q=q(T)
   * @param[in] d_reaction A function describing the derivative of the reaction dq = dq(T)/dT
   * @param[in] scale A scaling coefficient for the reaction term
   */
  void setNonlinearReaction(std::function<double(double)> reaction, std::function<double(double)> d_reaction,
                            std::unique_ptr<mfem::Coefficient>&& scale);

  /**
   * @brief Set the density field. Defaults to 1.0 if not set.
   *
   * @param[in] rho The density field coefficient
   */
  void setMassDensity(std::unique_ptr<mfem::Coefficient>&& rho);

  /**
   * @brief Set the specific heat capacity. Defaults to 1.0 if not set.
   *
   * @param[in] cp The specific heat capacity
   */
  void setSpecificHeatCapacity(std::unique_ptr<mfem::Coefficient>&& cp);

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
  virtual ~ThermalConductionLegacy() = default;

protected:
  /**
   * @brief The temperature finite element state
   */
  serac::FiniteElementState temperature_;

  /**
   * @brief Mass bilinear form object
   */
  std::unique_ptr<mfem::ParBilinearForm> M_form_;

  /**
   * @brief Stiffness nonlinear form object
   */
  std::unique_ptr<mfem::ParNonlinearForm> K_form_;

  /**
   * @brief Assembled mass matrix
   */
  std::unique_ptr<mfem::HypreParMatrix> M_;

  /**
   * @brief Conduction coefficient
   */
  std::unique_ptr<mfem::Coefficient> kappa_;

  /**
   * @brief Body source coefficient
   */
  std::unique_ptr<mfem::Coefficient> source_;

  /**
   * @brief Density coefficient
   *
   */
  std::unique_ptr<mfem::Coefficient> rho_;

  /**
   * @brief Specific heat capacity
   *
   */
  std::unique_ptr<mfem::Coefficient> cp_;

  /**
   * @brief Combined mass matrix coefficient (rho * cp)
   *
   */
  std::unique_ptr<mfem::Coefficient> mass_coef_;

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

  /**
   * @brief the nonlinear reaction function
   *
   */
  std::function<double(double)> reaction_;

  /**
   * @brief the derivative of the nonlinear reaction function
   *
   */
  std::function<double(double)> d_reaction_;

  /**
   * @brief a scaling factor for the reaction
   *
   */
  std::unique_ptr<mfem::Coefficient> reaction_scale_;
};

}  // namespace serac

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by inlet
 */
template <>
struct FromInlet<serac::ThermalConductionLegacy::InputOptions> {
  /// @brief Returns created object from Inlet container
  serac::ThermalConductionLegacy::InputOptions operator()(const axom::inlet::Container& base);
};

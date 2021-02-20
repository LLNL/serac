// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermal_solver.hpp
 *
 * @brief An object containing the solver for a thermal conduction PDE
 */

#pragma once

#include "mfem.hpp"

#include "serac/physics/base_physics.hpp"
#include "serac/physics/operators/odes.hpp"
#include "serac/physics/operators/stdfunction_operator.hpp"

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
class ThermalConduction : public BasePhysics {
public:
  /**
   * @brief A timestep method and config for the M solver
   */
  struct TimesteppingOptions {
    TimestepMethod             timestepper;
    DirichletEnforcementMethod enforcement_method;
  };

  /**
   * @brief A configuration variant for the various solves
   * Either quasistatic, or time-dependent with timestep and M options
   */
  struct SolverOptions {
    LinearSolverOptions                T_lin_options;
    NonlinearSolverOptions             T_nonlin_options;
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
     * @param[in] table Inlet's Table that input files will be added to
     **/
    static void defineInputFileSchema(axom::inlet::Table& table);

    // The order of the field
    int           order;
    SolverOptions solver_options;
    // Conductivity
    double kappa;
    double cp;
    double rho;

    // Nonlinear reaction information
    std::function<double(double)>                 reaction_func;
    std::function<double(double)>                 d_reaction_func;
    std::optional<input::CoefficientInputOptions> reaction_scale_coef;

    // Source information
    std::optional<input::CoefficientInputOptions> source_coef;

    // Boundary condition information
    std::unordered_map<std::string, input::BoundaryConditionInputOptions> boundary_conditions;

    // Initial conditions for temperature
    std::optional<input::CoefficientInputOptions> initial_temperature;
  };

  static IterativeSolverOptions defaultLinearOptions()
  {
    return {.rel_tol     = 1.0e-6,
            .abs_tol     = 1.0e-12,
            .print_level = 0,
            .max_iter    = 200,
            .lin_solver  = LinearSolver::CG,
            .prec        = HypreSmootherPrec{mfem::HypreSmoother::Jacobi}};
  }

  static NonlinearSolverOptions defaultNonlinearOptions()
  {
    return {.rel_tol = 1.0e-4, .abs_tol = 1.0e-8, .max_iter = 500, .print_level = 1};
  }

  static SolverOptions defaultQuasistaticOptions()
  {
    return {defaultLinearOptions(), defaultNonlinearOptions(), std::nullopt};
  }

  static SolverOptions defaultDynamicOptions()
  {
    return {defaultLinearOptions(), defaultNonlinearOptions(),
            TimesteppingOptions{TimestepMethod::BackwardEuler, DirichletEnforcementMethod::RateControl}};
  }

  /**
   * @brief Construct a new Thermal Solver object
   *
   * @param[in] order The order of the thermal field discretization
   * @param[in] mesh The MFEM parallel mesh to solve the PDE on
   * @param[in] options The system solver parameters
   */
  ThermalConduction(int order, std::shared_ptr<mfem::ParMesh> mesh, const SolverOptions& options);

  /**
   * @brief Construct a new Thermal Solver object
   *
   * @param[in] mesh The MFEM parallel mesh to solve the PDE on
   * @param[in] options The solver information parsed from the input file
   */
  ThermalConduction(std::shared_ptr<mfem::ParMesh> mesh, const InputOptions& options);

  /**
   * @brief Set essential temperature boundary conditions (strongly enforced)
   *
   * @param[in] temp_bdr The boundary attributes on which to enforce a temperature
   * @param[in] temp_bdr_coef The prescribed boundary temperature
   */
  void setTemperatureBCs(const std::set<int>& temp_bdr, std::shared_ptr<mfem::Coefficient> temp_bdr_coef);

  void setAdjointEssentialBCs(mfem::Coefficient& adjoint_bdr_coef);

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

  void solveAdjoint();

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
  serac::FiniteElementState&       temperature() { return temperature_; };

  const serac::FiniteElementState& adjoint() const { return adjoint_; };
  serac::FiniteElementState&       adjoint() { return adjoint_; };

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
  virtual ~ThermalConduction() = default;

protected:
  /**
   * @brief The temperature finite element state
   */
  serac::FiniteElementState temperature_;
  serac::FiniteElementState adjoint_;

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

  double       dt_, previous_dt_;
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

template <>
struct FromInlet<serac::ThermalConduction::InputOptions> {
  serac::ThermalConduction::InputOptions operator()(const axom::inlet::Table& base);
};

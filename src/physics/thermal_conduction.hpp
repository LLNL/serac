// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermal_solver.hpp
 *
 * @brief An object containing the solver for a thermal conduction PDE
 */

#ifndef THERMAL_CONDUCTION
#define THERMAL_CONDUCTION

#include "mfem.hpp"
#include "physics/base_physics.hpp"
#include "physics/operators/thermal_operators.hpp"

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
   * @brief A timestep method and configs for the M and T solvers (in that order)
   */
  using DynamicParameters = std::tuple<TimestepMethod, LinearSolverParameters, LinearSolverParameters>;
  /**
   * @brief A configuration variant for the various solves
   * Either quasistatic with a single config for the K solver, or
   * time-dependent with timestep and M/T params
   */
  using ThermalConductionParameters = std::variant<LinearSolverParameters, DynamicParameters>;

  /**
   * @brief Construct a new Thermal Solver object
   *
   * @param[in] order The order of the thermal field discretization
   * @param[in] mesh The MFEM parallel mesh to solve the PDE on
   * @param[in] solver The system solver instance
   */
  ThermalConduction(int order, std::shared_ptr<mfem::ParMesh> mesh,
                    const ThermalConduction::ThermalConductionParameters& params = default_quasistatic);

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
   * @brief Get the temperature state
   *
   * @return A pointer to the current temperature finite element state
   */
  std::shared_ptr<serac::FiniteElementState> temperature() { return temperature_; };

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

  /**
   * @brief The default parameters for an iterative linear solver
   */
  constexpr static IterativeSolverParameters default_linear_params = {
      .rel_tol     = 1.0e-6,
      .abs_tol     = 1.0e-12,
      .print_level = 0,
      .max_iter    = 100,
      .lin_solver  = LinearSolver::CG,
      .prec        = HypreSmootherPrec{mfem::HypreSmoother::Jacobi}};

  constexpr static ThermalConductionParameters default_quasistatic = default_linear_params;
  constexpr static ThermalConductionParameters default_explicit_solve =
      std::make_tuple(TimestepMethod::ForwardEuler, default_linear_params, default_linear_params);
  constexpr static ThermalConductionParameters default_implicit_solve =
      std::make_tuple(TimestepMethod::BackwardEuler, default_linear_params, default_linear_params);

protected:
  /**
   * @brief The temperature finite element state
   */
  std::shared_ptr<serac::FiniteElementState> temperature_;

  /**
   * @brief Mass bilinear form object
   */
  std::unique_ptr<mfem::ParBilinearForm> M_form_;

  /**
   * @brief Stiffness bilinear form object
   */
  std::unique_ptr<mfem::ParBilinearForm> K_form_;

  /**
   * @brief Assembled mass matrix
   */
  std::unique_ptr<mfem::HypreParMatrix> M_mat_;

  /**
   * @brief Assembled stiffness matrix
   */
  std::unique_ptr<mfem::HypreParMatrix> K_mat_;

  /**
   * @brief Thermal load linear form
   */
  std::unique_ptr<mfem::ParLinearForm> l_form_;

  /**
   * @brief Assembled BC load vector
   */
  std::unique_ptr<mfem::HypreParVector> bc_rhs_;

  /**
   * @brief Assembled RHS vector
   */
  std::unique_ptr<mfem::HypreParVector> rhs_;

  /**
   * @brief Conduction coefficient
   */
  std::unique_ptr<mfem::Coefficient> kappa_;

  /**
   * @brief Body source coefficient
   */
  std::unique_ptr<mfem::Coefficient> source_;

  /**
   * @brief Configuration for dynamic equation solvers
   */
  std::optional<std::pair<LinearSolverParameters, LinearSolverParameters>> dyn_oper_params_;

  /**
   * @brief Time integration operator
   */
  std::unique_ptr<DynamicConductionOperator> dyn_oper_;

  /**
   * @brief Solve the Quasi-static operator
   */
  void quasiStaticSolve();

  /**
   * @brief System solver instance for quasistatic K solver
   */
  std::optional<EquationSolver> K_inv_;
};

}  // namespace serac

#endif

// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file nonlinear_solid_solver.hpp
 *
 * @brief The solver object for finite deformation hyperelasticity
 */

#ifndef NONLIN_SOLID
#define NONLIN_SOLID

#include "mfem.hpp"
#include "physics/base_physics.hpp"
#include "physics/operators/nonlinear_solid_operators.hpp"

namespace serac {

/**
 * @brief The nonlinear solid solver class
 *
 * The nonlinear hyperelastic quasi-static and dynamic
 * hyperelastic solver object. It is derived from MFEM
 * example 10p.
 */
class NonlinearSolid : public BasePhysics {
public:
  /**
   * @brief A timestep method and config for the M solver (in that order)
   */
  using DynamicParameters = std::tuple<TimestepMethod, LinearSolverParameters>;
  /**
   * @brief A configuration variant for the various solves
   * Either quasistatic, or time-dependent with timestep and M params
   */
  using NonlinearSolidParameters =
      std::tuple<LinearSolverParameters, NonlinearSolverParameters, std::optional<DynamicParameters>>;

  /**
   * @brief Construct a new Nonlinear Solid Solver object
   *
   * @param[in] order The order of the displacement field
   * @param[in] mesh The MFEM parallel mesh to solve on
   * @param[in] solver The system solver instance
   */
  NonlinearSolid(int order, std::shared_ptr<mfem::ParMesh> mesh,
                 const NonlinearSolidParameters& params = default_quasistatic);

  /**
   * @brief Set displacement boundary conditions
   *
   * @param[in] disp_bdr The set of boundary attributes to set the displacement on
   * @param[in] disp_bdr_coef The vector coefficient containing the set displacement values
   */
  void setDisplacementBCs(const std::set<int>& disp_bdr, std::shared_ptr<mfem::VectorCoefficient> disp_bdr_coef);

  /**
   * @brief Set the displacement essential boundary conditions on a single component
   *
   * @param[in] disp_bdr The set of boundary attributes to set the displacement on
   * @param[in] disp_bdr_coef The vector coefficient containing the set displacement values
   * @param[in] component The component to set the displacment on
   */
  void setDisplacementBCs(const std::set<int>& disp_bdr, std::shared_ptr<mfem::Coefficient> disp_bdr_coef,
                          int component);

  /**
   * @brief Set the traction boundary conditions
   *
   * @param[in] trac_bdr The set of boundary attributes to apply a traction to
   * @param[in] trac_bdr_coef The vector valued traction coefficient
   * @param[in] component The component to apply the traction on
   */
  void setTractionBCs(const std::set<int>& trac_bdr, std::shared_ptr<mfem::VectorCoefficient> trac_bdr_coef,
                      int component = -1);

  /**
   * @brief Set the viscosity coefficient
   *
   * @param[in] visc_coef The abstract viscosity coefficient
   */
  void setViscosity(std::unique_ptr<mfem::Coefficient>&& visc_coef);

  /**
   * @brief Set the hyperelastic material parameters
   *
   * @param[in] mu Set the mu Lame parameter for the hyperelastic solid
   * @param[in] K Set the K Lame parameter for the hyperelastic solid
   */
  void setHyperelasticMaterialParameters(double mu, double K);

  /**
   * @brief Set the initial displacement value
   *
   * @param[in] disp_state The initial displacement state
   */
  void setDisplacement(mfem::VectorCoefficient& disp_state);

  /**
   * @brief Set the velocity state
   *
   * @param[in] velo_state The velocity state
   */
  void setVelocity(mfem::VectorCoefficient& velo_state);

  /**
   * @brief Get the displacement state
   *
   * @return The displacement state field
   */
  std::shared_ptr<FiniteElementState> displacement() { return displacement_; };

  /**
   * @brief Get the velocity state
   *
   * @return The velocity state field
   */
  std::shared_ptr<FiniteElementState> velocity() { return velocity_; };

  /**
   * @brief Complete the setup of all of the internal MFEM objects and prepare for timestepping
   */
  void completeSetup() override;

  /**
   * @brief Advance the timestep
   *
   * @param[inout] dt The timestep to attempt. This will return the actual timestep for adaptive timestepping schemes
   */
  void advanceTimestep(double& dt) override;

  /**
   * @brief Destroy the Nonlinear Solid Solver object
   */
  virtual ~NonlinearSolid();

  /**
   * @brief The default parameters for an iterative linear solver
   */
  constexpr static IterativeSolverParameters default_qs_linear_params = {
      .rel_tol     = 1.0e-6,
      .abs_tol     = 1.0e-8,
      .print_level = 0,
      .max_iter    = 5000,
      .lin_solver  = LinearSolver::MINRES,
      .prec        = HypreSmootherPrec{mfem::HypreSmoother::l1Jacobi}};

  /**
   * @brief The default parameters for the nonlinear Newton solver for quasistatic mode
   */
  constexpr static NonlinearSolverParameters default_qs_nonlinear_params = {
      .rel_tol = 1.0e-3, .abs_tol = 1.0e-6, .max_iter = 5000, .print_level = 1};

  /**
   * @brief The default equation solver parameters for quasistatic simulations
   */
  constexpr static NonlinearSolidParameters default_quasistatic =
      std::make_tuple(default_qs_linear_params, default_qs_nonlinear_params, std::nullopt);
  /**
   * @brief The default parameters for an iterative linear solver
   */
  constexpr static IterativeSolverParameters default_dyn_linear_params = {.rel_tol     = 1.0e-4,
                                                                          .abs_tol     = 1.0e-8,
                                                                          .print_level = 0,
                                                                          .max_iter    = 500,
                                                                          .lin_solver  = LinearSolver::GMRES,
                                                                          .prec        = HypreBoomerAMGPrec{}};

  /**
   * @brief The default parameters for the nonlinear solid dynamic operator (M solver)
   */
  constexpr static IterativeSolverParameters default_dyn_oper_linear_params = {.rel_tol     = 1.0e-4,
                                                                               .abs_tol     = 1.0e-8,
                                                                               .print_level = 0,
                                                                               .max_iter    = 500,
                                                                               .lin_solver  = LinearSolver::GMRES,
                                                                               .prec        = HypreSmootherPrec{}};

  /**
   * @brief The default parameters for the nonlinear Newton solver for dynamic mode
   */
  constexpr static NonlinearSolverParameters default_dyn_nonlinear_params = {
      .rel_tol = 1.0e-4, .abs_tol = 1.0e-8, .max_iter = 500, .print_level = 1};

  /**
   * @brief The default equation solver parameters for time-dependent simulations
   */
  constexpr static NonlinearSolidParameters default_dynamic =
      std::make_tuple(default_dyn_linear_params, default_dyn_nonlinear_params,
                      std::make_tuple(TimestepMethod::SDIRK33, default_dyn_oper_linear_params));

protected:
  /**
   * @brief Velocity field
   */
  std::shared_ptr<FiniteElementState> velocity_;

  /**
   * @brief Displacement field
   */
  std::shared_ptr<FiniteElementState> displacement_;

  /**
   * @brief The quasi-static operator for use with the MFEM newton solvers
   */
  std::unique_ptr<mfem::Operator> nonlinear_oper_;

  /**
   * @brief Configuration for dynamic equation solver
   */
  std::optional<LinearSolverParameters> timedep_oper_params_;

  /**
   * @brief The time dependent operator for use with the MFEM ODE solvers
   */
  std::unique_ptr<mfem::TimeDependentOperator> timedep_oper_;

  /**
   * @brief The viscosity coefficient
   */
  std::unique_ptr<mfem::Coefficient> viscosity_;

  /**
   * @brief The hyperelastic material model
   */
  std::unique_ptr<mfem::HyperelasticModel> model_;

  /**
   * @brief Pointer to the reference mesh data
   */
  std::unique_ptr<mfem::ParGridFunction> reference_nodes_;

  /**
   * @brief Pointer to the deformed mesh data
   */
  std::unique_ptr<mfem::ParGridFunction> deformed_nodes_;

  /**
   * @brief Solve the Quasi-static operator
   */
  void quasiStaticSolve();

  /**
   * @brief Nonlinear system solver instance
   */
  EquationSolver nonlin_solver_;
};

}  // namespace serac

#endif

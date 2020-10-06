// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file nonlinear_solid_operators.hpp
 *
 * @brief The operators that form the nonlinear solid PDE solver
 */

#ifndef NONLINSOLID_OPER
#define NONLINSOLID_OPER

#include <memory>

#include "mfem.hpp"
#include "physics/utilities/boundary_condition_manager.hpp"
#include "physics/utilities/equation_solver.hpp"
#include "physics/utilities/solver_config.hpp"

namespace serac {

struct NonlinearSolidBC {
  struct Displacement : VectorBoundaryCondition {
  };
  struct Traction : VectorBoundaryCondition {
  };
  using DisplacementEss = StrongAlias<EssentialBoundaryCondition, Displacement>;
  using TractionNat     = StrongAlias<NaturalBoundaryCondition, Traction>;
  using Manager         = BoundaryConditionManager<DisplacementEss, TractionNat>;
};

/**
 * @brief The abstract MFEM operator for a quasi-static solve
 */
class NonlinearSolidQuasiStaticOperator : public mfem::Operator {
public:
  /**
   * @brief Construct a new Nonlinear Solid Quasi Static Operator object
   *
   * @param[in] H_form The nonlinear form of the PDE
   */
  explicit NonlinearSolidQuasiStaticOperator(std::unique_ptr<mfem::ParNonlinearForm> H_form,
                                             const NonlinearSolidBC::Manager&        bcs);

  /**
   * @brief Get the Gradient of the nonlinear form
   *
   * @param[in] x The input state to evaluate the gradient
   * @return The global gradient operator of the nonlinear form at state x
   */
  mfem::Operator& GetGradient(const mfem::Vector& x) const;

  /**
   * @brief Residual evaluation of the nonlinear form
   *
   * @param[in] k The input statue to evalue the residual
   * @param[out] y The output residual of the nonlinear form
   */
  void Mult(const mfem::Vector& k, mfem::Vector& y) const;

  /**
   * @brief Destroy the Nonlinear Solid Quasi Static Operator object
   */
  virtual ~NonlinearSolidQuasiStaticOperator();

protected:
  /**
   * @brief The nonlinear form
   */
  std::unique_ptr<mfem::ParNonlinearForm> H_form_;

  /**
   * @brief The linearized jacobian at the current state
   */
  mutable std::unique_ptr<mfem::Operator> Jacobian_;

  /**
   * @brief The boundary conditions
   */
  const NonlinearSolidBC::Manager& bcs_;
};

/**
 * @brief The reduced dynamic solid operator
 *
 *  Nonlinear operator of the form:
 *  k --> (M + dt*S)*k + H(x + dt*v + dt^2*k) + S*v,
 *  where M and S are given BilinearForms, H is a given NonlinearForm, v and x
 *  are given vectors, and dt is a scalar.S
 */
class NonlinearSolidReducedSystemOperator : public mfem::Operator {
public:
  /**
   * @brief Construct a new Nonlinear Solid Reduced System Operator object
   *
   * @param[in] H_form The nonlinear stiffness form
   * @param[in] S_form The linear viscosity form
   * @param[in] M_form The linear mass form
   * @param[in] ess_bdr The essential boundary conditions
   */
  NonlinearSolidReducedSystemOperator(const mfem::ParNonlinearForm& H_form, const mfem::ParBilinearForm& S_form,
                                      mfem::ParBilinearForm& M_form, const NonlinearSolidBC::Manager& bcs);

  /**
   * @brief Set current dt, v, x values - needed to compute action and Jacobian.
   *
   * @param[in] dt The current timestep
   * @param[in] v The current velocity
   * @param[in] x The current position
   */
  void SetParameters(double dt, const mfem::Vector* v, const mfem::Vector* x);

  /**
   * @brief Compute y = H(x + dt (v + dt k)) + M k + S (v + dt k).
   *
   * @param[in] k The input state to evaluate the residual
   * @param[out] y The output state to evaluate the residual
   */
  virtual void Mult(const mfem::Vector& k, mfem::Vector& y) const;

  /**
   * @brief Compute J = M + dt S + dt^2 grad_H(x + dt (v + dt k)).
   *
   * @param[in] k The input state
   * @return The gradient operator
   */
  virtual mfem::Operator& GetGradient(const mfem::Vector& k) const;

  /**
   * @brief Destroy the Nonlinear Solid Reduced System Operator object
   *
   */
  virtual ~NonlinearSolidReducedSystemOperator();

private:
  /**
   * @brief The bilinear form for the mass matrix
   */
  mfem::ParBilinearForm& M_form_;

  /**
   * @brief The bilinear form for the viscous terms
   */
  const mfem::ParBilinearForm& S_form_;

  /**
   * @brief The nonlinear form for the hyperelastic response
   */
  const mfem::ParNonlinearForm& H_form_;

  /**
   * @brief The linearized jacobian
   */
  mutable std::unique_ptr<mfem::HypreParMatrix> jacobian_;

  /**
   * @brief The current timestep
   */
  double dt_;

  /**
   * @brief The current velocity and displacement vectors
   */
  const mfem::Vector *v_, *x_;

  /**
   * @brief Working vectors
   */
  mutable mfem::Vector w_, z_;

  /**
   * @brief Essential degrees of freedom
   */
  const NonlinearSolidBC::Manager& bcs_;
};

/**
 * @brief The abstract time dependent MFEM operator for explicit and implicit solves
 */
class NonlinearSolidDynamicOperator : public mfem::TimeDependentOperator {
public:
  /**
   * @brief Construct a new Nonlinear Solid Dynamic Operator object
   *
   * @param[in] H_form The nonlinear stiffness form
   * @param[in] S_form The linear viscosity form
   * @param[in] M_form The linear mass form
   * @param[in] ess_bdr The essential boundary conditions
   * @param[in] newton_solver The newton solver object
   * @param[in] lin_params The linear solver parameters
   */
  NonlinearSolidDynamicOperator(std::unique_ptr<mfem::ParNonlinearForm> H_form,
                                std::unique_ptr<mfem::ParBilinearForm>  S_form,
                                std::unique_ptr<mfem::ParBilinearForm> M_form, const NonlinearSolidBC::Manager& bcs,
                                EquationSolver& newton_solver, const serac::LinearSolverParameters& lin_params);

  /**
   * @brief Evaluate the explicit time derivative
   *
   * @param[in] vx The current velocity and displacement state
   * @param[out] dvx_dt The explicit time derivative of the state vector
   */
  virtual void Mult(const mfem::Vector& vx, mfem::Vector& dvx_dt) const;

  /**
   * @brief Solve the Backward-Euler equation: k = f(x + dt*k, t), for the unknown k.
   *
   * This is the only requirement for high-order SDIRK implicit integration.
   *
   * @param[in] dt The timestep
   * @param[in] vx The state block vector of velocities and displacements
   * @param[out] dvx_dt The time rate of vx
   */
  virtual void ImplicitSolve(const double dt, const mfem::Vector& vx, mfem::Vector& dvx_dt);

  /**
   * @brief Destroy the Nonlinear Solid Dynamic Operator object
   */
  virtual ~NonlinearSolidDynamicOperator();

protected:
  /**
   * @brief The bilinear form for the mass matrix
   */
  std::unique_ptr<mfem::ParBilinearForm> M_form_;

  /**
   * @brief The bilinear form for the viscous terms
   */
  std::unique_ptr<mfem::ParBilinearForm> S_form_;

  /**
   * @brief The nonlinear form for the hyperelastic response
   */
  std::unique_ptr<mfem::ParNonlinearForm> H_form_;

  /**
   * @brief The assembled mass matrix
   */
  std::unique_ptr<mfem::HypreParMatrix> M_mat_;

  /**
   * @brief The CG solver for the mass matrix
   */
  EquationSolver M_inv_;

  /**
   * @brief The reduced system operator for applying the bilinear and nonlinear forms
   */
  std::unique_ptr<NonlinearSolidReducedSystemOperator> reduced_oper_;

  /**
   * @brief The Newton solver for the nonlinear iterations
   */
  EquationSolver& newton_solver_;

  /**
   * @brief The fixed boudnary degrees of freedom
   */
  const NonlinearSolidBC::Manager& bcs_;

  /**
   * @brief The linear solver parameters for the mass matrix
   */
  serac::LinearSolverParameters lin_params_;

  /**
   * @brief Working vector
   */
  mutable mfem::Vector z_;
};

}  // namespace serac

#endif

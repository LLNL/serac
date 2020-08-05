// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef NONLINSOLID_OPER
#define NONLINSOLID_OPER

#include <memory>

#include "common/serac_types.hpp"
#include "mfem.hpp"

namespace serac {

/**
 * The abstract MFEM operator for a quasi-static solve
 */
class NonlinearSolidQuasiStaticOperator : public mfem::Operator {
 protected:
  /**
   * The nonlinear form
   */
  std::shared_ptr<mfem::ParNonlinearForm> H_form_;

  /**
   * The linearized jacobian at the current state
   */
  mutable std::unique_ptr<mfem::Operator> Jacobian_;

 public:
  /**
   * The constructor
   */
  explicit NonlinearSolidQuasiStaticOperator(std::shared_ptr<mfem::ParNonlinearForm> H_form);

  /**
   * Required to use the native newton solver
   */
  mfem::Operator& GetGradient(const mfem::Vector& x) const;

  /**
   * Required for residual calculations
   */
  void Mult(const mfem::Vector& k, mfem::Vector& y) const;

  /**
   * The destructor
   */
  virtual ~NonlinearSolidQuasiStaticOperator();
};

/**
 *  Nonlinear operator of the form:
 *  k --> (M + dt*S)*k + H(x + dt*v + dt^2*k) + S*v,
 *  where M and S are given BilinearForms, H is a given NonlinearForm, v and x
 *  are given vectors, and dt is a scalar.S
 */
class NonlinearSolidReducedSystemOperator : public mfem::Operator {
 private:
  /**
   * The bilinear form for the mass matrix
   */
  mfem::ParBilinearForm& M_form_;

  /**
   * The bilinear form for the viscous terms
   */
  const mfem::ParBilinearForm& S_form_;

  /**
   * The nonlinear form for the hyperelastic response
   */
  const mfem::ParNonlinearForm& H_form_;

  /**
   * The linearized jacobian
   */
  mutable std::unique_ptr<mfem::HypreParMatrix> jacobian_;

  /**
   * The current timestep
   */
  double dt_;

  /**
   * The current displacement and velocity vectors
   */
  const mfem::Vector *v_, *x_;

  /**
   * Working vectors
   */
  mutable mfem::Vector w_, z_;

  /**
   * Essential degrees of freedom
   */
  const std::vector<std::shared_ptr<serac::BoundaryCondition> >& ess_bdr_;

 public:
  /**
   * The constructor
   */
  NonlinearSolidReducedSystemOperator(const mfem::ParNonlinearForm& H_form, const mfem::ParBilinearForm& S_form,
                                      mfem::ParBilinearForm&                                         M_form,
                                      const std::vector<std::shared_ptr<serac::BoundaryCondition> >& ess_bdr);

  /**
   * Set current dt, v, x values - needed to compute action and Jacobian.
   */
  void SetParameters(double dt, const mfem::Vector* v, const mfem::Vector* x);

  /**
   * Compute y = H(x + dt (v + dt k)) + M k + S (v + dt k).
   */
  virtual void Mult(const mfem::Vector& k, mfem::Vector& y) const;

  /**
   * Compute J = M + dt S + dt^2 grad_H(x + dt (v + dt k)).
   */
  virtual mfem::Operator& GetGradient(const mfem::Vector& k) const;

  /**
   * The destructor
   */
  virtual ~NonlinearSolidReducedSystemOperator();
};

/**
 * The abstract time dependent MFEM operator for explicit and implicit solves
 */
class NonlinearSolidDynamicOperator : public mfem::TimeDependentOperator {
 protected:
  /**
   * The bilinear form for the mass matrix
   */
  std::unique_ptr<mfem::ParBilinearForm> M_form_;

  /**
   * The bilinear form for the viscous terms
   */
  std::unique_ptr<mfem::ParBilinearForm> S_form_;

  /**
   * The nonlinear form for the hyperelastic response
   */
  std::unique_ptr<mfem::ParNonlinearForm> H_form_;

  /**
   * The assembled mass matrix
   */
  std::unique_ptr<mfem::HypreParMatrix> M_mat_;

  /**
   * The CG solver for the mass matrix
   */
  mfem::CGSolver M_solver_;

  /**
   * The preconditioner for the CG mass matrix solver
   */
  mfem::HypreSmoother M_prec_;

  /**
   * The reduced system operator for applying the bilinear and nonlinear forms
   */
  std::unique_ptr<NonlinearSolidReducedSystemOperator> reduced_oper_;

  /**
   * The Newton solver for the nonlinear iterations
   */
  mfem::NewtonSolver& newton_solver_;

  /**
   * The fixed boudnary degrees of freedom
   */
  const std::vector<std::shared_ptr<serac::BoundaryCondition> >& ess_bdr_;

  /**
   * The linear solver parameters for the mass matrix
   */
  serac::LinearSolverParameters lin_params_;

  /**
   * Working vector
   */
  mutable mfem::Vector z_;

 public:
  /**
   * The constructor
   */
  NonlinearSolidDynamicOperator(std::unique_ptr<mfem::ParNonlinearForm>                        H_form,
                                std::unique_ptr<mfem::ParBilinearForm>                         S_form,
                                std::unique_ptr<mfem::ParBilinearForm>                         M_form,
                                const std::vector<std::shared_ptr<serac::BoundaryCondition> >& ess_bdr,
                                mfem::NewtonSolver& newton_solver, const serac::LinearSolverParameters& lin_params);

  /**
   * Required to use the native newton solver
   */
  virtual void Mult(const mfem::Vector& vx, mfem::Vector& dvx_dt) const;

  /**
   * Solve the Backward-Euler equation: k = f(x + dt*k, t), for the unknown k.
   * This is the only requirement for high-order SDIRK implicit integration.
   */
  virtual void ImplicitSolve(const double dt, const mfem::Vector& x, mfem::Vector& k);

  /**
   * The destructor
   */
  virtual ~NonlinearSolidDynamicOperator();
};

}  // namespace serac

#endif

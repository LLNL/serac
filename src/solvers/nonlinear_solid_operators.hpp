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

/// The abstract MFEM operator for a quasi-static solve
class NonlinearSolidQuasiStaticOperator : public mfem::Operator {
 protected:
  /// The nonlinear form
  std::shared_ptr<mfem::ParNonlinearForm> m_H_form;

  /// The linearized jacobian at the current state
  mutable std::unique_ptr<mfem::Operator> m_Jacobian;

 public:
  /// The constructor
  NonlinearSolidQuasiStaticOperator(std::shared_ptr<mfem::ParNonlinearForm> H_form);

  /// Required to use the native newton solver
  mfem::Operator &GetGradient(const mfem::Vector &x) const;

  /// Required for residual calculations
  void Mult(const mfem::Vector &k, mfem::Vector &y) const;

  /// The destructor
  virtual ~NonlinearSolidQuasiStaticOperator();
};

///  Nonlinear operator of the form:
///  k --> (M + dt*S)*k + H(x + dt*v + dt^2*k) + S*v,
///  where M and S are given BilinearForms, H is a given NonlinearForm, v and x
///  are given vectors, and dt is a scalar.S
class NonlinearSolidReducedSystemOperator : public mfem::Operator {
 private:
  /// The bilinear form for the mass matrix
  std::shared_ptr<mfem::ParBilinearForm> m_M_form;

  /// The bilinear form for the viscous terms
  std::shared_ptr<mfem::ParBilinearForm> m_S_form;

  /// The nonlinear form for the hyperelastic response
  std::shared_ptr<mfem::ParNonlinearForm> m_H_form;

  /// The linearized jacobian
  mutable std::unique_ptr<mfem::HypreParMatrix> m_jacobian;

  /// The current timestep
  double m_dt;

  /// The current displacement and velocity vectors
  const mfem::Vector *m_v, *m_x;

  /// Working vectors
  mutable mfem::Vector m_w, m_z;

  /// Essential degrees of freedom
  const std::vector<std::shared_ptr<serac::BoundaryCondition> > &m_ess_bdr;

 public:
  /// The constructor
  NonlinearSolidReducedSystemOperator(std::shared_ptr<mfem::ParNonlinearForm>                 H_form,
                                      std::shared_ptr<mfem::ParBilinearForm>                  S_form,
                                      std::shared_ptr<mfem::ParBilinearForm>                  M_form,
                                      const std::vector<std::shared_ptr<serac::BoundaryCondition> > &ess_bdr);

  /// Set current dt, v, x values - needed to compute action and Jacobian.
  void SetParameters(double dt, const mfem::Vector *v, const mfem::Vector *x);

  /// Compute y = H(x + dt (v + dt k)) + M k + S (v + dt k).
  virtual void Mult(const mfem::Vector &k, mfem::Vector &y) const;

  /// Compute J = M + dt S + dt^2 grad_H(x + dt (v + dt k)).
  virtual mfem::Operator &GetGradient(const mfem::Vector &k) const;

  /// The destructor
  virtual ~NonlinearSolidReducedSystemOperator();
};

/// The abstract time dependent MFEM operator for explicit and implicit solves
class NonlinearSolidDynamicOperator : public mfem::TimeDependentOperator {
 protected:
  /// The bilinear form for the mass matrix
  std::shared_ptr<mfem::ParBilinearForm> m_M_form;

  /// The bilinear form for the viscous terms
  std::shared_ptr<mfem::ParBilinearForm> m_S_form;

  /// The nonlinear form for the hyperelastic response
  std::shared_ptr<mfem::ParNonlinearForm> m_H_form;

  /// The assembled mass matrix
  std::unique_ptr<mfem::HypreParMatrix> m_M_mat;

  /// The CG solver for the mass matrix
  mfem::CGSolver m_M_solver;

  /// The preconditioner for the CG mass matrix solver
  mfem::HypreSmoother m_M_prec;

  /// The reduced system operator for applying the bilinear and nonlinear forms
  std::unique_ptr<NonlinearSolidReducedSystemOperator> m_reduced_oper;

  /// The Newton solver for the nonlinear iterations
  mfem::NewtonSolver &m_newton_solver;

  /// The fixed boudnary degrees of freedom
  const std::vector<std::shared_ptr<serac::BoundaryCondition> > &m_ess_bdr;

  /// The linear solver parameters for the mass matrix
  serac::LinearSolverParameters m_lin_params;

  /// Working vector
  mutable mfem::Vector m_z;

 public:
  /// The constructor
  NonlinearSolidDynamicOperator(std::shared_ptr<mfem::ParNonlinearForm>                 H_form,
                                std::shared_ptr<mfem::ParBilinearForm>                  S_form,
                                std::shared_ptr<mfem::ParBilinearForm>                  M_form,
                                const std::vector<std::shared_ptr<serac::BoundaryCondition> > &ess_bdr,
                                mfem::NewtonSolver &newton_solver, const serac::LinearSolverParameters &lin_params);

  /// Required to use the native newton solver
  virtual void Mult(const mfem::Vector &vx, mfem::Vector &dvx_dt) const;

  /// Solve the Backward-Euler equation: k = f(x + dt*k, t), for the unknown k.
  /// This is the only requirement for high-order SDIRK implicit integration.
  virtual void ImplicitSolve(const double dt, const mfem::Vector &x, mfem::Vector &k);

  /// The destructor
  virtual ~NonlinearSolidDynamicOperator();
};

#endif

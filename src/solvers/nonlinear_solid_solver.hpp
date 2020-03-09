// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef NONLINSOLID_SOLVER
#define NONLINSOLID_SOLVER

#include "mfem.hpp"
#include "base_solver.hpp"

// Forward declaration
class NonlinearSolidQuasiStaticOperator;
class NonlinearSolidDynamicOperator;
class NonlinearSolidReducedSystemOperator;

class NonlinearSolidSolver : public BaseSolver
{
protected:
  /// The abstract nonlinear form
  mfem::ParNonlinearForm *m_H_form;

  /// The abstract mass bilinear form
  mfem::ParBilinearForm *m_S_form;

  /// The abstract viscosity bilinear form
  mfem::ParBilinearForm *m_S_form;

  /// The operator for use with the MFEM newton and ODE solvers
  mfem::Operator *m_nonlinear_oper;

  /// The Newton solver for the nonlinear iterations
  mfem::NewtonSolver m_newton_solver;

  /// The linear solver for the Jacobian
  mfem::Solver *m_J_solver;

  /// The preconditioner for the Jacobian solver
  mfem::Solver *m_J_prec;

  /// The viscosity coefficient
  mfem::Coefficient &m_viscosity;

  /// The hyperelastic material model
  mfem::HyperelasticModel *m_model;

  /// Linear solver parameters
  LinearSolverParameters m_lin_params;

  /// Solve the Quasi-static operator
  void QuasiStaticSolve();

public:
  /// Constructor from order and parallel mesh
  NonlinearSolidSolver(int order, mfem::ParMesh *pmesh);

  /// Set the displacement essential boundary conditions
  void SetDisplacementBCs(mfem::Array<int> &disp_bdr, mfem::VectorCoefficient *disp_bdr_coef);

  /// Set the traction boundary conditions
  void SetTractionBCs(mfem::Array<int> &trac_bdr, mfem::VectorCoefficient *trac_bdr_coef);

  /// Set the hyperelastic material parameters
  void SetHyperelasticMaterialParameters(double mu, double K);

  /// Set the initial state (guess)
  void SetInitialState(mfem::VectorCoefficient &state);

  /// Set the linear solver params
  void SetLinearSolverParameters(const LinearSolverParameters &params);

  /// Complete the data structure initialization
  void CompleteSetup();

  /// Advance the timestep
  void AdvanceTimestep(double &dt);

  /// Destructor
  virtual ~NonlinearSolidSolver();
};


class NonlinearSolidQuasiStaticOperator : public mfem::Operator
{
protected:
  mfem::ParNonlinearForm *m_H_form;

  mutable mfem::Operator *m_Jacobian;

public:
  NonlinearSolidQuasiStaticOperator(mfem::ParNonlinearForm *H_form);
      
  /// Required to use the native newton solver
  mfem::Operator &GetGradient(const mfem::Vector &x) const;

  void Mult(const mfem::Vector &k, mfem::Vector &y) const;

  virtual ~NonlinearSolidQuasiStaticOperator();
};

class NonlinearSolidDynamicOperator : public mfem::TimeDependentOperator
{
protected:
  mfem::ParBilinearForm *m_M_form;
  mfem::ParBilinearForm *m_S_form;
  mfem::ParNonlinearForm *m_H_form;

  mfem::HypreParMatrix *m_M_mat; // Mass matrix from ParallelAssemble()
  mfem::CGSolver m_M_solver;    // Krylov solver for inverting the mass matrix M
  mfem::HypreSmoother m_M_prec; // Preconditioner for the mass matrix M

  NonlinearSolidReducedSystemOperator *m_reduced_oper;

  LinearSolverParameters m_lin_params;

public:
  NonlinearSolidDynamicOperator(mfem::ParNonlinearForm *H_form, mfem::ParBilinearForm *S_form, mfem::ParBilinearForm *M_form, 
                                const mfem::Array<int> &ess_tdof_list, LinearSolverParameters lin_params);

  /// Required to use the native newton solver
  virtual void Mult(const mfem::Vector &vx, mfem::Vector &dvx_dt) const;
  /// Solve the Backward-Euler equation: k = f(x + dt*k, t), for the unknown k.
  /// This is the only requirement for high-order SDIRK implicit integration.
  virtual void ImplicitSolve(const double dt, const mfem::Vector &x, mfem::Vector &k);

  virtual ~NonlinearSolidQuasiStaticOperator();
};

//  Nonlinear operator of the form:
//  k --> (M + dt*S)*k + H(x + dt*v + dt^2*k) + S*v,
//  where M and S are given BilinearForms, H is a given NonlinearForm, v and x
//  are given vectors, and dt is a scalar.
class NonlinearSolidReducedSystemOperator : public mfem::Operator
{
private:
  mfem::ParBilinearForm *m_M_form;
  mfem::ParBilinearForm *m_S_form;
  mfem::ParNonlinearForm *m_H_form;
  mutable mfem::HypreParMatrix *m_jacobian;
  double m_dt;
  const mfem::Vector *m_v, *m_x;
  mutable mfem::Vector m_w, m_z;
  const mfem::Array<int> &m_ess_tdof_list;

public:
  NonlinearSolidReducedSystemOperator(mfem::ParNonlinearForm *H_form, mfem::ParBilinearForm *S_form,
    mfem::ParBilinearForm *M_form, const mfem::Array<int> &ess_tdof_list);

  /// Set current dt, v, x values - needed to compute action and Jacobian.
  void SetParameters(double dt, const mfem::Vector *v, const mfem::Vector *x);

  /// Compute y = H(x + dt (v + dt k)) + M k + S (v + dt k).
  virtual void Mult(const mfem::Vector &k, mfem::Vector &y) const;

  /// Compute J = M + dt S + dt^2 grad_H(x + dt (v + dt k)).
  virtual mfem::Operator &GetGradient(const mfem::Vector &k) const;

  virtual ~NonlinearSolidReducedSystemOperator();
};

#endif

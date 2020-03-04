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

class NonlinearSolidSolver : public BaseSolver
{
protected:
  /// The abstract nonlinear form
  mfem::ParNonlinearForm *m_H_form;

  /// The operator for use with the MFEM newton and ODE solvers
  mfem::Operator *m_nonlinear_oper;

  /// The Newton solver for the nonlinear iterations
  mfem::NewtonSolver m_newton_solver;

  /// The linear solver for the Jacobian
  mfem::Solver *m_J_solver;

  /// The preconditioner for the Jacobian solver
  mfem::Solver *m_J_prec;

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

#endif

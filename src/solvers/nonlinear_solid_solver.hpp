// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef NONLINSOLID_SOLVER
#define NONLINSOLID_SOLVER

#include "base_solver.hpp"
#include "mfem.hpp"
#include "nonlinear_solid_operators.hpp"

/// The nonlinear hyperelastic quasi-static and dynamic
/// hyperelastic solver object. It is derived from MFEM
/// example 10p.
class NonlinearSolidSolver : public BaseSolver {
 protected:
  FiniteElementState &velocity;
  FiniteElementState &displacement;

  /// The abstract nonlinear form
  std::shared_ptr<mfem::ParNonlinearForm> m_H_form;

  /// The abstract mass bilinear form
  std::shared_ptr<mfem::ParBilinearForm> m_M_form;

  /// The abstract viscosity bilinear form
  std::shared_ptr<mfem::ParBilinearForm> m_S_form;

  /// The quasi-static operator for use with the MFEM newton solvers
  std::shared_ptr<mfem::Operator> m_nonlinear_oper;

  /// The time dependent operator for use with the MFEM ODE solvers
  std::shared_ptr<mfem::TimeDependentOperator> m_timedep_oper;

  /// The Newton solver for the nonlinear iterations
  mfem::NewtonSolver m_newton_solver;

  /// The linear solver for the Jacobian
  std::shared_ptr<mfem::Solver> m_J_solver;

  /// The preconditioner for the Jacobian solver
  std::shared_ptr<mfem::Solver> m_J_prec;

  /// The viscosity coefficient
  std::shared_ptr<mfem::Coefficient> m_viscosity;

  /// The hyperelastic material model
  std::shared_ptr<mfem::HyperelasticModel> m_model;

  /// Linear solver parameters
  LinearSolverParameters m_lin_params;

  /// Nonlinear solver parameters
  NonlinearSolverParameters m_nonlin_params;

  /// Solve the Quasi-static operator
  void QuasiStaticSolve();

 public:
  /// Constructor from order and parallel mesh
  NonlinearSolidSolver(int order, const std::shared_ptr<mfem::ParMesh> &pmesh);

  /// Set the displacement essential boundary conditions
  void SetDisplacementBCs(const std::vector<int> &                        disp_bdr,
                          const std::shared_ptr<mfem::VectorCoefficient> &disp_bdr_coef);

  /// Set the traction boundary conditions
  void SetTractionBCs(const std::vector<int> &trac_bdr, const std::shared_ptr<mfem::VectorCoefficient> &trac_bdr_coef);

  /// Set the viscosity coefficient
  void SetViscosity(const std::shared_ptr<mfem::Coefficient> &visc_coef);

  /// Set the hyperelastic material parameters
  void SetHyperelasticMaterialParameters(double mu, double K);

  /// Set the initial state (guess)
  void SetInitialState(mfem::VectorCoefficient &disp_state, mfem::VectorCoefficient &velo_state);

  /// Set the linear and nonlinear solver params
  void SetSolverParameters(const LinearSolverParameters &lin_params, const NonlinearSolverParameters &nonlin_params);

  /// Complete the data structure initialization
  void CompleteSetup();

  /// Advance the timestep
  void AdvanceTimestep(double &dt);

  /// Destructor
  virtual ~NonlinearSolidSolver();
};

#endif

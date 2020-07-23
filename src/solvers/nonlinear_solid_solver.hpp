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
  std::shared_ptr<serac::FiniteElementState> m_velocity;
  std::shared_ptr<serac::FiniteElementState> m_displacement;

  /// The quasi-static operator for use with the MFEM newton solvers
  std::shared_ptr<mfem::Operator> m_nonlinear_oper;

  /// The time dependent operator for use with the MFEM ODE solvers
  std::shared_ptr<mfem::TimeDependentOperator> m_timedep_oper;

  /// The Newton solver for the nonlinear iterations
  mfem::NewtonSolver m_newton_solver;

  /// The linear solver for the Jacobian
  std::unique_ptr<mfem::Solver> m_J_solver;

  /// The preconditioner for the Jacobian solver
  std::unique_ptr<mfem::Solver> m_J_prec;

  /// The viscosity coefficient
  std::shared_ptr<mfem::Coefficient> m_viscosity;

  /// The hyperelastic material model
  std::shared_ptr<mfem::HyperelasticModel> m_model;

  /// Linear solver parameters
  serac::LinearSolverParameters m_lin_params;

  /// Nonlinear solver parameters
  serac::NonlinearSolverParameters m_nonlin_params;

  /// Pointer to the reference mesh data
  std::unique_ptr<mfem::ParGridFunction> m_reference_nodes;

  /// Pointer to the deformed mesh data
  std::unique_ptr<mfem::ParGridFunction> m_deformed_nodes;

  /// Solve the Quasi-static operator
  void QuasiStaticSolve();

 public:
  /// Constructor from order and parallel mesh
  NonlinearSolidSolver(int order, std::shared_ptr<mfem::ParMesh> pmesh);

  /// Set the displacement essential boundary conditions
  void SetDisplacementBCs(const std::set<int> &disp_bdr, std::shared_ptr<mfem::VectorCoefficient> disp_bdr_coef);

  /// Set the displacement essential boundary conditions on a single component
  void SetDisplacementBCs(const std::set<int> &disp_bdr, std::shared_ptr<mfem::Coefficient> disp_bdr_coef,
                          int component);

  /// Set the traction boundary conditions
  void SetTractionBCs(const std::set<int> &trac_bdr, std::shared_ptr<mfem::VectorCoefficient> trac_bdr_coef,
                      int component = -1);

  /// Set the viscosity coefficient
  void SetViscosity(std::shared_ptr<mfem::Coefficient> visc_coef);

  /// Set the hyperelastic material parameters
  void SetHyperelasticMaterialParameters(double mu, double K);

  /// Set the initial displacement state (guess)
  void SetDisplacement(mfem::VectorCoefficient &disp_state);

  /// Set the initial velocity state (guess)
  void SetVelocity(mfem::VectorCoefficient &velo_state);

  /// Set the linear and nonlinear solver params
  void SetSolverParameters(const serac::LinearSolverParameters &lin_params, const serac::NonlinearSolverParameters &nonlin_params);

  /// Get the displacement state
  std::shared_ptr<serac::FiniteElementState> GetDisplacement() { return m_displacement; };

  /// Get the velocity state
  std::shared_ptr<serac::FiniteElementState> GetVelocity() { return m_velocity; };

  /// Complete the data structure initialization
  void CompleteSetup() override;

  /// Advance the timestep
  void AdvanceTimestep(double &dt) override;

  /// Destructor
  virtual ~NonlinearSolidSolver();
};

#endif

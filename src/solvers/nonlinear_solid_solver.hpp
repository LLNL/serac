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

namespace serac {

/// The nonlinear hyperelastic quasi-static and dynamic
/// hyperelastic solver object. It is derived from MFEM
/// example 10p.
class NonlinearSolidSolver : public BaseSolver {
 protected:
  std::shared_ptr<serac::FiniteElementState> velocity_;
  std::shared_ptr<serac::FiniteElementState> displacement_;

  /// The abstract nonlinear form
  std::shared_ptr<mfem::ParNonlinearForm> H_form_;

  /// The abstract mass bilinear form
  std::shared_ptr<mfem::ParBilinearForm> M_form_;

  /// The abstract viscosity bilinear form
  std::shared_ptr<mfem::ParBilinearForm> S_form_;

  /// The quasi-static operator for use with the MFEM newton solvers
  std::shared_ptr<mfem::Operator> nonlinear_oper_;

  /// The time dependent operator for use with the MFEM ODE solvers
  std::shared_ptr<mfem::TimeDependentOperator> timedep_oper_;

  /// The Newton solver for the nonlinear iterations
  mfem::NewtonSolver newton_solver_;

  /// The linear solver for the Jacobian
  std::shared_ptr<mfem::Solver> J_solver_;

  /// The preconditioner for the Jacobian solver
  std::shared_ptr<mfem::Solver> J_prec_;

  /// The viscosity coefficient
  std::shared_ptr<mfem::Coefficient> viscosity_;

  /// The hyperelastic material model
  std::shared_ptr<mfem::HyperelasticModel> model_;

  /// Linear solver parameters
  serac::LinearSolverParameters lin_params_;

  /// Nonlinear solver parameters
  serac::NonlinearSolverParameters nonlin_params_;

  /// Pointer to the reference mesh data
  std::unique_ptr<mfem::ParGridFunction> reference_nodes_;

  /// Pointer to the deformed mesh data
  std::unique_ptr<mfem::ParGridFunction> deformed_nodes_;

  /// Solve the Quasi-static operator
  void quasiStaticSolve();

 public:
  /// Constructor from order and parallel mesh
  NonlinearSolidSolver(int order, std::shared_ptr<mfem::ParMesh> pmesh);

  /// Set the displacement essential boundary conditions
  void setDisplacementBCs(const std::set<int>& disp_bdr, std::shared_ptr<mfem::VectorCoefficient> disp_bdr_coef);

  /// Set the displacement essential boundary conditions on a single component
  void setDisplacementBCs(const std::set<int>& disp_bdr, std::shared_ptr<mfem::Coefficient> disp_bdr_coef,
                          int component);

  /// Set the traction boundary conditions
  void setTractionBCs(const std::set<int>& trac_bdr, std::shared_ptr<mfem::VectorCoefficient> trac_bdr_coef,
                      int component = -1);

  /// Set the viscosity coefficient
  void setViscosity(std::shared_ptr<mfem::Coefficient> visc_coef);

  /// Set the hyperelastic material parameters
  void setHyperelasticMaterialParameters(double mu, double K);

  /// Set the initial displacement state (guess)
  void setDisplacement(mfem::VectorCoefficient& disp_state);

  /// Set the initial velocity state (guess)
  void setVelocity(mfem::VectorCoefficient& velo_state);

  /// Set the linear and nonlinear solver params
  void setSolverParameters(const serac::LinearSolverParameters&    lin_params,
                           const serac::NonlinearSolverParameters& nonlin_params);

  /// Get the displacement state
  std::shared_ptr<serac::FiniteElementState> getDisplacement() { return displacement_; };

  /// Get the velocity state
  std::shared_ptr<serac::FiniteElementState> getVelocity() { return velocity_; };

  /// Complete the data structure initialization
  void completeSetup();

  /// Advance the timestep
  void advanceTimestep(double& dt);

  /// Destructor
  virtual ~NonlinearSolidSolver();
};

}  // namespace serac

#endif

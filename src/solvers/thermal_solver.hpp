// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef CONDUCTION_SOLVER
#define CONDUCTION_SOLVER

#include "base_solver.hpp"
#include "mfem.hpp"
#include "thermal_operators.hpp"

/** This is a generic linear thermal diffusion oeprator of the form
 *
 *    M du/dt = -kappa Ku + f
 *
 *  where M is a mass matrix, K is a stiffness matrix, and f is a
 *  thermal load vector. */
class ThermalSolver : public BaseSolver {
 protected:
  std::shared_ptr<serac::FiniteElementState> temperature_;

  /// Mass bilinear form object
  std::unique_ptr<mfem::ParBilinearForm> M_form_;

  /// Stiffness bilinear form object
  std::unique_ptr<mfem::ParBilinearForm> K_form_;

  /// Assembled mass matrix
  std::shared_ptr<mfem::HypreParMatrix> M_mat_;

  /// Eliminated mass matrix
  std::shared_ptr<mfem::HypreParMatrix> M_e_mat_;

  /// Assembled stiffness matrix
  std::shared_ptr<mfem::HypreParMatrix> K_mat_;

  /// Eliminated stiffness matrix
  std::shared_ptr<mfem::HypreParMatrix> K_e_mat_;

  /// Thermal load linear form
  std::unique_ptr<mfem::ParLinearForm> l_form_;

  /// Assembled BC load vector
  std::shared_ptr<mfem::HypreParVector> bc_rhs_;

  /// Assembled RHS vector
  std::shared_ptr<mfem::HypreParVector> rhs_;

  /// Linear solver for the K operator
  std::shared_ptr<mfem::CGSolver> K_solver_;

  /// Preconditioner for the K operator
  std::shared_ptr<mfem::HypreSmoother> K_prec_;

  /// Conduction coefficient
  std::shared_ptr<mfem::Coefficient> kappa_;

  /// Body source coefficient
  std::shared_ptr<mfem::Coefficient> source_;

  /// Time integration operator
  std::unique_ptr<DynamicConductionOperator> dyn_oper_;

  /// Linear solver parameters
  serac::LinearSolverParameters lin_params_;

  /// Solve the Quasi-static operator
  void QuasiStaticSolve();

 public:
  /// Constructor from order and parallel mesh
  ThermalSolver(int order, std::shared_ptr<mfem::ParMesh> pmesh);

  /// Set essential temperature boundary conditions (strongly enforced)
  void setTemperatureBCs(const std::set<int> &temp_bdr, std::shared_ptr<mfem::Coefficient> temp_bdr_coef);

  /// Set flux boundary conditions (weakly enforced)
  void setFluxBCs(const std::set<int> &flux_bdr, std::shared_ptr<mfem::Coefficient> flux_bdr_coef);

  /// Advance the timestep using the chosen integration scheme
  void advanceTimestep(double &dt);

  /// Set the thermal conductivity coefficient
  void setConductivity(std::shared_ptr<mfem::Coefficient> kappa);

  /// Set the temperature from a coefficient
  void setTemperature(mfem::Coefficient &temp);

  /// Set the body thermal source from a coefficient
  void setSource(std::shared_ptr<mfem::Coefficient> source);

  /// Get the temperature state
  std::shared_ptr<serac::FiniteElementState> getTemperature() { return temperature_; };

  /** Complete the initialization and allocation of the data structures. This
   *  must be called before StaticSolve() or AdvanceTimestep(). If allow_dynamic
   * = false, do not allocate the mass matrix or dynamic operator */
  void completeSetup();

  /// Set the linear solver parameters for both the M and K operators
  void setLinearSolverParameters(const serac::LinearSolverParameters &params);

  /// Destructor
  virtual ~ThermalSolver() = default;
};

#endif

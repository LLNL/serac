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
  std::shared_ptr<FiniteElementState> m_temperature;

  /// Mass bilinear form object
  std::unique_ptr<mfem::ParBilinearForm> m_M_form;

  /// Stiffness bilinear form object
  std::unique_ptr<mfem::ParBilinearForm> m_K_form;

  /// Assembled mass matrix
  std::shared_ptr<mfem::HypreParMatrix> m_M_mat;

  /// Eliminated mass matrix
  std::shared_ptr<mfem::HypreParMatrix> m_M_e_mat;

  /// Assembled stiffness matrix
  std::shared_ptr<mfem::HypreParMatrix> m_K_mat;

  /// Eliminated stiffness matrix
  std::shared_ptr<mfem::HypreParMatrix> m_K_e_mat;

  /// Thermal load linear form
  std::unique_ptr<mfem::ParLinearForm> m_l_form;

  /// Assembled BC load vector
  std::shared_ptr<mfem::HypreParVector> m_bc_rhs;

  /// Assembled RHS vector
  std::shared_ptr<mfem::HypreParVector> m_rhs;

  /// Linear solver for the K operator
  std::shared_ptr<mfem::CGSolver> m_K_solver;

  /// Preconditioner for the K operator
  std::shared_ptr<mfem::HypreSmoother> m_K_prec;

  /// Conduction coefficient
  std::shared_ptr<mfem::Coefficient> m_kappa;

  /// Body source coefficient
  std::shared_ptr<mfem::Coefficient> m_source;

  /// Time integration operator
  std::unique_ptr<DynamicConductionOperator> m_dyn_oper;

  /// Linear solver parameters
  LinearSolverParameters m_lin_params;

  /// Solve the Quasi-static operator
  void QuasiStaticSolve();

 public:
  /// Constructor from order and parallel mesh
  ThermalSolver(int order, std::shared_ptr<mfem::ParMesh> pmesh);

  /// Set essential temperature boundary conditions (strongly enforced)
  void SetTemperatureBCs(const std::vector<int> &temp_bdr, std::shared_ptr<mfem::Coefficient> temp_bdr_coef);

  /// Set flux boundary conditions (weakly enforced)
  void SetFluxBCs(const std::vector<int> &flux_bdr, std::shared_ptr<mfem::Coefficient> flux_bdr_coef);

  /// Advance the timestep using the chosen integration scheme
  void AdvanceTimestep(double &dt);

  /// Set the thermal conductivity coefficient
  void SetConductivity(std::shared_ptr<mfem::Coefficient> kappa);

  /// Set the temperature from a coefficient
  void SetTemperature(mfem::Coefficient &temp);

  /// Set the body thermal source from a coefficient
  void SetSource(std::shared_ptr<mfem::Coefficient> source);

  /// Get the temperature state
  std::shared_ptr<FiniteElementState> GetTemperature() {return m_temperature;};

  /** Complete the initialization and allocation of the data structures. This
   *  must be called before StaticSolve() or AdvanceTimestep(). If allow_dynamic
   * = false, do not allocate the mass matrix or dynamic operator */
  void CompleteSetup();

  /// Set the linear solver parameters for both the M and K operators
  void SetLinearSolverParameters(const LinearSolverParameters &params);

  /// Destructor
  virtual ~ThermalSolver() = default;
};

#endif

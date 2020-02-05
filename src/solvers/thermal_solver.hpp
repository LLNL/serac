// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef CONDUCTION_SOLVER
#define CONDUCTION_SOLVER

#include "mfem.hpp"
#include "base_solver.hpp"

// Forward declaration
class DynamicConductionOperator;

/** This is a generic linear thermal diffusion oeprator of the form
 *
 *    M du/dt = -kappa Ku + f
 *
 *  where M is a mass matrix, K is a stiffness matrix, and f is a
 *  thermal load vector. */
class ThermalSolver : public BaseSolver
{
protected:
  /// Mass bilinear form object
  mfem::ParBilinearForm *m_M_form;

  /// Stiffness bilinear form object
  mfem::ParBilinearForm *m_K_form;

  /// Assembled mass matrix
  mfem::HypreParMatrix *m_M_mat;

  /// Eliminated mass matrix
  mfem::HypreParMatrix *m_M_e_mat;

  /// Assembled stiffness matrix
  mfem::HypreParMatrix *m_K_mat;

  /// Eliminated stiffness matrix
  mfem::HypreParMatrix *m_K_e_mat;

  /// Thermal load linear form
  mfem::ParLinearForm *m_l_form;

  /// Assembled BC load vector
  mfem::HypreParVector *m_bc_rhs;

  /// Assembled RHS vector
  mfem::HypreParVector *m_rhs;

  /// Linear solver for the K operator
  mfem::CGSolver *m_K_solver;

  /// Preconditioner for the K operator
  mfem::HypreSmoother *m_K_prec;

  /// Conduction coefficient
  mfem::Coefficient *m_kappa;

  /// Body source coefficient
  mfem::Coefficient *m_source;

  /// Time integration operator
  DynamicConductionOperator *m_dyn_oper;

  /// Linear solver parameters
  LinearSolverParameters m_lin_params;

  /// Solve the Quasi-static operator
  void QuasiStaticSolve();

public:
  /// Constructor from order and parallel mesh
  ThermalSolver(int order, mfem::ParMesh *pmesh);

  /// Set essential temperature boundary conditions (strongly enforced)
  void SetTemperatureBCs(mfem::Array<int> &temp_bdr, mfem::Coefficient *temp_bdr_coef);

  /// Set flux boundary conditions (weakly enforced)
  void SetFluxBCs(mfem::Array<int> &flux_bdr, mfem::Coefficient *flux_bdr_coef);

  /// Advance the timestep using the chosen integration scheme
  void AdvanceTimestep(double &dt);

  /// Set the thermal conductivity coefficient
  void SetConductivity(mfem::Coefficient &kappa);

  /// Set the initial temperature from a coefficient
  void SetInitialState(mfem::Coefficient &temp);

  /// Set the body thermal source from a coefficient
  void SetSource(mfem::Coefficient &source);

  /** Complete the initialization and allocation of the data structures. This
   *  must be called before StaticSolve() or AdvanceTimestep(). If allow_dynamic = false,
   *  do not allocate the mass matrix or dynamic operator */
  void CompleteSetup();

  /// Set the linear solver parameters for both the M and K operators
  void SetLinearSolverParameters(const LinearSolverParameters &params);

  /// Destructor
  virtual ~ThermalSolver();
};

/// The time dependent operator for advancing the discretized conduction ODE
class DynamicConductionOperator : public mfem::TimeDependentOperator
{
protected:

  /// Finite Element space
  mfem::ParFiniteElementSpace *m_fespace;

  /// Grid function for boundary condition projection
  mfem::ParGridFunction *m_state_gf;

  /// Solver for the mass matrix
  mfem::CGSolver *m_M_solver;

  /// Solver for the T matrix
  mfem::CGSolver *m_T_solver;

  /// Preconditioner for the M matrix
  mfem::HypreSmoother *m_M_prec;

  /// Preconditioner for the T matrix
  mfem::HypreSmoother *m_T_prec;

  /// Pointer to the assembled M matrix
  mfem::HypreParMatrix *m_M_mat;

  /// Pointer to the eliminated M matrix
  mfem::HypreParMatrix *m_M_e_mat;

  /// Pointer to the assembled K matrix
  mfem::HypreParMatrix *m_K_mat;

  /// Pointer to the eliminated K matrix
  mfem::HypreParMatrix *m_K_e_mat;

  /// Pointer to the assembled T ( = M + dt K) matrix
  mfem::HypreParMatrix *m_T_mat;

  /// Pointer to the eliminated T matrix
  mfem::HypreParMatrix *m_T_e_mat;

  /// Assembled RHS vector
  mfem::Vector *m_rhs;

  /// RHS vector including essential boundary elimination
  mfem::Vector *m_bc_rhs;

  /// Temperature essential boundary coefficient
  mfem::Coefficient *m_ess_bdr_coef;

  /// Essential temperature boundary markers
  mutable mfem::Array<int> m_ess_bdr;

  /// Essential true DOFs
  mfem::Array<int> m_ess_tdof_list;

  /// Auxillary working vectors
  mutable mfem::Vector m_z;
  mutable mfem::Vector m_y;
  mutable mfem::Vector m_x;

  /// Storage of old dt use to determine if we should recompute the T matrix
  mutable double m_old_dt;

public:
  /// Constructor. Height is the true degree of freedom size
  DynamicConductionOperator(mfem::ParFiniteElementSpace *fespace, LinearSolverParameters &params);

  /// Set the mass matrix
  void SetMMatrix(mfem::HypreParMatrix *M_mat, mfem::HypreParMatrix *M_e_mat);

  /// Set the stiffness matrix
  void SetKMatrix(mfem::HypreParMatrix *K_mat, mfem::HypreParMatrix *K_e_mat);

  /// Set the load vector
  void SetLoadVector(mfem::Vector *rhs);

  /// Set the essential temperature boundary information
  void SetEssentialBCs(mfem::Coefficient *ess_bdr_coef, mfem::Array<int> &ess_bdr, mfem::Array<int> &ess_tdof_list);

  /** Calculate du_dt = M^-1 (-Ku + f).
   *  This is all that is needed for explicit methods */
  virtual void Mult(const mfem::Vector &u, mfem::Vector &du_dt) const;

  /** Solve the Backward-Euler equation: du_dt = M^-1[-K(u + dt * du_dt)]
   *  for du_dt. This is needed for implicit methods */
  virtual void ImplicitSolve(const double dt, const mfem::Vector &u, mfem::Vector &du_dt);

  ///Destructor
  virtual ~DynamicConductionOperator();

};


#endif

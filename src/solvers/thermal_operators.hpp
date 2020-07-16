// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef CONDUCTION_OPER
#define CONDUCTION_OPER

#include <memory>

#include "common/serac_types.hpp"
#include "mfem.hpp"

/// The time dependent operator for advancing the discretized conduction ODE
class DynamicConductionOperator : public mfem::TimeDependentOperator {
 protected:
  /// Finite Element space
  std::shared_ptr<mfem::ParFiniteElementSpace> m_fespace;

  /// Grid function for boundary condition projection
  std::shared_ptr<mfem::ParGridFunction> m_state_gf;

  /// Solver for the mass matrix
  std::unique_ptr<mfem::CGSolver> m_M_solver;

  /// Solver for the T matrix
  std::unique_ptr<mfem::CGSolver> m_T_solver;

  /// Preconditioner for the M matrix
  std::unique_ptr<mfem::HypreSmoother> m_M_prec;

  /// Preconditioner for the T matrix
  std::unique_ptr<mfem::HypreSmoother> m_T_prec;

  /// Pointer to the assembled M matrix
  std::shared_ptr<mfem::HypreParMatrix> m_M_mat;

  /// Pointer to the assembled K matrix
  std::shared_ptr<mfem::HypreParMatrix> m_K_mat;

  /// Pointer to the assembled T ( = M + dt K) matrix
  std::unique_ptr<mfem::HypreParMatrix> m_T_mat;

  /// Pointer to the eliminated T matrix
  std::unique_ptr<mfem::HypreParMatrix> m_T_e_mat;

  /// Assembled RHS vector
  std::shared_ptr<mfem::Vector> m_rhs;

  /// RHS vector including essential boundary elimination
  std::shared_ptr<mfem::Vector> m_bc_rhs;

  /// Temperature essential boundary coefficient
  std::vector<std::shared_ptr<serac::BoundaryCondition> > m_ess_bdr;

  /// Auxillary working vectors
  mutable mfem::Vector m_z;
  mutable mfem::Vector m_y;
  mutable mfem::Vector m_x;

  /// Storage of old dt use to determine if we should recompute the T matrix
  mutable double m_old_dt;

 public:
  /// Constructor. Height is the true degree of freedom size
  DynamicConductionOperator(std::shared_ptr<mfem::ParFiniteElementSpace> fespace, const serac::LinearSolverParameters &params,
                            const std::vector<std::shared_ptr<serac::BoundaryCondition> > &ess_bdr);

  /// Set the mass matrix
  void SetMatrices(std::shared_ptr<mfem::HypreParMatrix> M_mat, std::shared_ptr<mfem::HypreParMatrix> K_mat);

  /// Set the load vector
  void SetLoadVector(std::shared_ptr<mfem::Vector> rhs);

  /** Calculate du_dt = M^-1 (-Ku + f).
   *  This is all that is needed for explicit methods */
  virtual void Mult(const mfem::Vector &u, mfem::Vector &du_dt) const;

  /** Solve the Backward-Euler equation: du_dt = M^-1[-K(u + dt * du_dt)]
   *  for du_dt. This is needed for implicit methods */
  virtual void ImplicitSolve(const double dt, const mfem::Vector &u, mfem::Vector &du_dt);

  /// Destructor
  virtual ~DynamicConductionOperator();
};

#endif
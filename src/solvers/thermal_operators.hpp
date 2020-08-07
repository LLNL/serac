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

namespace serac {

/**
 * The time dependent operator for advancing the discretized conduction ODE
 */
class DynamicConductionOperator : public mfem::TimeDependentOperator {
 protected:
  /**
   * Finite Element space
   */
  std::shared_ptr<mfem::ParFiniteElementSpace> fespace_;

  /**
   * Grid function for boundary condition projection
   */
  std::shared_ptr<mfem::ParGridFunction> state_gf_;

  /**
   * Solver for the mass matrix
   */
  std::unique_ptr<mfem::CGSolver> M_solver_;

  /**
   * Solver for the T matrix
   */
  std::unique_ptr<mfem::CGSolver> T_solver_;

  /**
   * Preconditioner for the M matrix
   */
  std::unique_ptr<mfem::HypreSmoother> M_prec_;

  /**
   * Preconditioner for the T matrix
   */
  std::unique_ptr<mfem::HypreSmoother> T_prec_;

  /**
   * Pointer to the assembled M matrix
   */
  std::shared_ptr<mfem::HypreParMatrix> M_mat_;

  /**
   * Pointer to the assembled K matrix
   */
  std::shared_ptr<mfem::HypreParMatrix> K_mat_;

  /**
   * Pointer to the assembled T ( = M + dt K) matrix
   */
  std::unique_ptr<mfem::HypreParMatrix> T_mat_;

  /**
   * Pointer to the eliminated T matrix
   */
  std::unique_ptr<mfem::HypreParMatrix> T_e_mat_;

  /**
   * Assembled RHS vector
   */
  std::shared_ptr<mfem::Vector> rhs_;

  /**
   * RHS vector including essential boundary elimination
   */
  std::shared_ptr<mfem::Vector> bc_rhs_;

  /**
   * Temperature essential boundary coefficient
   */
  std::vector<serac::BoundaryCondition>& ess_bdr_;

  /**
   * Auxillary working vectors
   */
  mutable mfem::Vector z_;
  mutable mfem::Vector y_;
  mutable mfem::Vector x_;

  /**
   * Storage of old dt use to determine if we should recompute the T matrix
   */
  mutable double old_dt_;

 public:
  /**
   * Constructor. Height is the true degree of freedom size
   */
  DynamicConductionOperator(std::shared_ptr<mfem::ParFiniteElementSpace> fespace,
                            const serac::LinearSolverParameters&         params,
                            std::vector<serac::BoundaryCondition>&       ess_bdr);

  /**
   * Set the mass matrix
   */
  void setMatrices(std::shared_ptr<mfem::HypreParMatrix> M_mat, std::shared_ptr<mfem::HypreParMatrix> K_mat);

  /**
   * Set the load vector
   */
  void setLoadVector(std::shared_ptr<mfem::Vector> rhs);

  /** 
   * Calculate du_dt = M^-1 (-Ku + f).
   * This is all that is needed for explicit methods
   */
  virtual void Mult(const mfem::Vector& u, mfem::Vector& du_dt) const;

  /** 
   * Solve the Backward-Euler equation: du_dt = M^-1[-K(u + dt * du_dt)]
   * for du_dt. This is needed for implicit methods
   */
  virtual void ImplicitSolve(const double dt, const mfem::Vector& u, mfem::Vector& du_dt);

  /**
   * Destructor
   */
  virtual ~DynamicConductionOperator();
};

}  // namespace serac

#endif

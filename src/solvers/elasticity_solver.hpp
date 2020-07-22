// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef LINEARELASTIC_SOLVER
#define LINEARELASTIC_SOLVER

#include "base_solver.hpp"
#include "mfem.hpp"

/** This is a generic linear elasticity oeprator of the form
 *
 *    -div(sigma(u)) = f
 *    sigma(u) = lambda div(u) + mu(grad(u) + grad(u)^T
 *
 *  where u is the displacement vector, f is the body force,
 *  and lambda and mu are the lame parameters */
class ElasticitySolver : public BaseSolver {
 protected:
  std::shared_ptr<FiniteElementState> displacement;

  /// Stiffness bilinear form
  std::unique_ptr<mfem::ParBilinearForm> m_K_form;

  /// Load bilinear form
  std::unique_ptr<mfem::ParLinearForm> m_l_form;

  /// Stiffness matrix
  std::unique_ptr<mfem::HypreParMatrix> m_K_mat;

  /// Eliminated stiffness matrix
  std::unique_ptr<mfem::HypreParMatrix> m_K_e_mat;

  /// RHS vector
  std::unique_ptr<mfem::HypreParVector> m_rhs;

  /// Eliminated RHS vector
  std::unique_ptr<mfem::HypreParVector> m_bc_rhs;

  /// Solver for the stiffness matrix
  std::unique_ptr<mfem::Solver> m_K_solver;

  /// Preconditioner for the stiffness
  std::unique_ptr<mfem::Solver> m_K_prec;

  /// Lame mu parameter coefficient
  mfem::Coefficient *m_mu;

  /// Lame lambda parameter coefficient
  mfem::Coefficient *m_lambda;

  /// Body source coefficient
  mfem::VectorCoefficient *m_body_force;

  /// Linear solver parameters
  LinearSolverParameters m_lin_params;

  /// Driver for a quasi-static solve
  void QuasiStaticSolve();

 public:
  /// Constructor using order and mesh
  ElasticitySolver(int order, std::shared_ptr<mfem::ParMesh> pmesh);

  /// Set the vector-valued essential displacement boundary conditions
  void SetDisplacementBCs(std::set<int> &disp_bdr, std::shared_ptr<mfem::VectorCoefficient> disp_bdr_coef,
                          int component = -1);

  /// Set the vector-valued natural traction boundary conditions
  void SetTractionBCs(std::set<int> &trac_bdr, std::shared_ptr<mfem::VectorCoefficient> trac_bdr_coef,
                      int component = -1);

  /// Driver for advancing the timestep
  void AdvanceTimestep(double &dt);

  /// Set the elastic lame parameters
  void SetLameParameters(mfem::Coefficient &lambda, mfem::Coefficient &mu);

  /// Set the vector-valued body force coefficient
  void SetBodyForce(mfem::VectorCoefficient &force);

  /// Finish the setup and allocate the associate data structures
  void CompleteSetup();

  /// Set the linear solver parameters object
  void SetLinearSolverParameters(const LinearSolverParameters &params);

  /// The destructor
  virtual ~ElasticitySolver();
};

#endif

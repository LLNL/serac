// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef LINEARELASTIC_SOLVER
#define LINEARELASTIC_SOLVER

#include "mfem.hpp"
#include "base_solver.hpp"

/** This is a generic linear elasticity oeprator of the form
 *
 *    -div(sigma(u)) = f
 *    sigma(u) = lambda div(u) + mu(grad(u) + grad(u)^T
 *
 *  where u is the displacement vector, f is the body force,
 *  and lambda and mu are the lame parameters */
class ElasticitySolver : public BaseSolver
{
protected:
  
  FiniteElementState & displacement;

  /// Stiffness bilinear form
  mfem::ParBilinearForm *m_K_form;

  /// Load bilinear form
  mfem::ParLinearForm *m_l_form;

  /// Stiffness matrix
  mfem::HypreParMatrix *m_K_mat;

  /// Eliminated stiffness matrix
  mfem::HypreParMatrix *m_K_e_mat;

  /// RHS vector
  mfem::HypreParVector *m_rhs;

  /// Eliminated RHS vector
  mfem::HypreParVector *m_bc_rhs;

  /// Solver for the stiffness matrix
  mfem::Solver *m_K_solver;

  /// Preconditioner for the stiffness
  mfem::Solver *m_K_prec;

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
  ElasticitySolver(int order, mfem::ParMesh *pmesh);

  /// Set the vector-valued essential displacement boundary conditions
  void SetDisplacementBCs(mfem::Array<int> &disp_bdr, mfem::VectorCoefficient *disp_bdr_coef);

  /// Set the vector-valued natural traction boundary conditions
  void SetTractionBCs(mfem::Array<int> &trac_bdr, mfem::VectorCoefficient *trac_bdr_coef);

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

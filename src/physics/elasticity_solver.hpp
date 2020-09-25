// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file elasticity_solver.hpp
 *
 * @brief A solver for the steady state solution of a linear elasticity PDE
 */

#ifndef LINEARELASTIC_SOLVER
#define LINEARELASTIC_SOLVER

#include "mfem.hpp"
#include "physics/base_solver.hpp"

namespace serac {

/**
 * @brief A solver for the steady state solution of a linear elasticity PDE
 *
 * This is a generic linear elasticity oeprator of the form
 *
 *    -div(sigma(u)) = f
 *    sigma(u) = lambda div(u) + mu(grad(u) + grad(u)^T
 *
 *  where u is the displacement vector, f is the body force,
 *  and lambda and mu are the lame parameters
 */
class ElasticitySolver : public BaseSolver {
public:
  /**
   * @brief Construct a new Elasticity Solver object
   *
   * @param[in] order The polynomial order of the solver
   * @param[in] mesh The parallel MFEM mesh
   */
  ElasticitySolver(const int order, std::shared_ptr<mfem::ParMesh> mesh);

  /**
   * @brief Set the vector-valued essential displacement boundary conditions
   *
   * @param[in] disp_bdr Set of boundary attributes to enforce the displacement conditions
   * @param[in] disp_bdr_coef Coefficient definining the displacement boundary
   * @param[in] component Component to set (-1 indicates all components)
   */
  void setDisplacementBCs(const std::set<int>& disp_bdr, mfem::VectorCoefficient& disp_bdr_coef,
                          const int component = -1);

  /**
   * @brief Set the vector-valued natural traction boundary conditions
   *
   * @param[in] trac_bdr Set of boundary attributes to enforce the traction condition
   * @param[in] trac_bdr_coef The traction condition coefficient
   * @param[in] component Component to set (-1 indicates all components)
   */
  void setTractionBCs(const std::set<int>& trac_bdr, mfem::VectorCoefficient& trac_bdr_coef, const int component = -1);

  /**
   * @brief Driver for advancing the timestep
   *
   * @param[inout] dt The timestep to attempt, adaptive methods could return the actual timestep completed
   */
  void advanceTimestep(double& dt) override;

  /**
   * @brief Set the elastic lame parameters
   *
   * @param[in] lambda The Lame lambda coefficient
   * @param[in] mu The Lame mu coefficient
   */
  void setLameParameters(mfem::Coefficient& lambda, mfem::Coefficient& mu);

  /**
   * @brief Set the Vector-valued body force
   *
   * @param[in] force The body force coefficient
   */
  void setBodyForce(mfem::VectorCoefficient& force);

  /**
   * @brief Finish the setup of the solver and allocate and initialize the associated MFEM data structures
   */
  void completeSetup() override;

  /**
   * @brief Set the Linear Solver Parameters
   *
   * @param[in] params The linear solver parameters
   */
  void setLinearSolverParameters(const serac::LinearSolverParameters& params);

  /**
   * @brief The destructor
   */
  virtual ~ElasticitySolver();

protected:
  /**
   * @brief Displacement field
   */
  std::shared_ptr<serac::FiniteElementState> displacement_;

  /**
   * @brief Stiffness bilinear form
   */
  std::unique_ptr<mfem::ParBilinearForm> K_form_;

  /**
   * @brief Load bilinear form
   */
  std::unique_ptr<mfem::ParLinearForm> l_form_;

  /**
   * @brief Stiffness matrix
   */
  std::unique_ptr<mfem::HypreParMatrix> K_mat_;

  /**
   * @brief RHS load vector
   */
  std::unique_ptr<mfem::HypreParVector> rhs_;

  /**
   * @brief Eliminated RHS load vector
   */
  std::unique_ptr<mfem::HypreParVector> bc_rhs_;

  /**
   * @brief Linear solver for the stiffness matrix
   */
  std::unique_ptr<mfem::Solver> K_solver_;

  /**
   * @brief Preconditioner for the stiffness matrix
   */
  std::unique_ptr<mfem::Solver> K_prec_;

  /**
   * @brief Lame mu elasticity parameter
   */
  mfem::Coefficient* mu_ = nullptr;

  /**
   * @brief Lame lambda elasticity parameter
   */
  mfem::Coefficient* lambda_ = nullptr;

  /**
   * @brief Body force coefficient
   */
  mfem::VectorCoefficient* body_force_ = nullptr;

  /**
   * @brief Linear solver parameters
   */
  serac::LinearSolverParameters lin_params_;

  /**
   * @brief Quasi-static solve driver
   */
  void QuasiStaticSolve();
};

}  // namespace serac

#endif

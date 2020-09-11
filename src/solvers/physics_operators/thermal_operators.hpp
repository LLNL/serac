// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermal_operators.hpp
 *
 * @brief The MFEM oeprators for the thermal conduction solver
 */

#ifndef CONDUCTION_OPER
#define CONDUCTION_OPER

#include <memory>

#include "mfem.hpp"
#include "solvers/physics_utilities/boundary_condition.hpp"
#include "solvers/physics_utilities/equation_solver.hpp"
#include "solvers/physics_utilities/solver_config.hpp"

namespace serac {

/**
 * @brief The time dependent operator for advancing the discretized conduction ODE
 */
class DynamicConductionOperator : public mfem::TimeDependentOperator {
public:
  /**
   * @brief Construct a new Dynamic Conduction Operator object
   *
   * @param[in] fe_space The temperature field finite element space
   * @param[in] params The linear solver parameters
   * @param[in] ess_bdr The essential boundary condition objects
   */
  DynamicConductionOperator(mfem::ParFiniteElementSpace& fe_space, const serac::LinearSolverParameters& params,
                            std::vector<serac::BoundaryCondition>& ess_bdr);

  /**
   * @brief Set the mass and stiffness matrices
   *
   * @param[in] M_mat The mass matrix
   * @param[in] K_mat The stiffness matrix
   */
  void setMatrices(const mfem::HypreParMatrix* M_mat, mfem::HypreParMatrix* K_mat);

  /**
   * @brief Set the thermal flux load vector
   *
   * @param[in] rhs The thermal flux (RHS vector)
   */
  void setLoadVector(const mfem::Vector* rhs);

  /**
   * @brief Calculate du_dt = M^-1 (-Ku + f)
   *
   * This is all that is required for explicit time integration methods
   *
   * @param[in] u The input state vector
   * @param[out] du_dt The output time derivative of the state vector
   */
  virtual void Mult(const mfem::Vector& u, mfem::Vector& du_dt) const;

  /**
   * @brief Solve the Backward-Euler equation: du_dt = M^-1[-K(u + dt * du_dt)]
   *
   * This is required for implicit time integration schemes
   *
   * @param[in] dt The timestep
   * @param[in] u The input state vector
   * @param[out] du_dt The output time derivative of the state vector
   */
  virtual void ImplicitSolve(const double dt, const mfem::Vector& u, mfem::Vector& du_dt);

  /**
   * @brief Destroy the Dynamic Conduction Operator object
   */
  virtual ~DynamicConductionOperator();

protected:
  /**
   * @brief Grid function for boundary condition projection
   */
  std::unique_ptr<mfem::ParGridFunction> state_gf_;

  /**
   * @brief Solver for the mass matrix
   */
  EquationSolver M_inv_;

  /**
   * @brief Solver for the T matrix
   */
  EquationSolver T_inv_;

  /**
   * @brief Non-owning pointer to the assembled M matrix
   */
  const mfem::HypreParMatrix* M_ = nullptr;

  /**
   * @brief Non-owning pointer to the assembled K matrix
   */
  mfem::HypreParMatrix* K_ = nullptr;

  /**
   * @brief Pointer to the assembled T ( = M + dt K) matrix
   */
  std::unique_ptr<mfem::HypreParMatrix> T_;

  /**
   * @brief Pointer to the eliminated T matrix
   */
  std::unique_ptr<mfem::HypreParMatrix> T_e_mat_;

  /**
   * @brief Non-owning ptr to assembled RHS vector
   */
  const mfem::Vector* rhs_ = nullptr;

  /**
   * @brief RHS vector including essential boundary elimination
   */
  std::unique_ptr<mfem::Vector> bc_rhs_;

  /**
   * @brief Temperature essential boundary coefficient
   */
  std::vector<serac::BoundaryCondition>& ess_bdr_;

  /**
   * @brief Auxillary working vectors
   */
  mutable mfem::Vector z_;
  mutable mfem::Vector y_;
  mutable mfem::Vector x_;

  /**
   * @brief Storage of old dt use to determine if we should recompute the T matrix
   */
  mutable double old_dt_;
};

}  // namespace serac

#endif

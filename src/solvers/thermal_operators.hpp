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

#include "common/serac_types.hpp"
#include "mfem.hpp"

namespace serac {

/**
 * @brief The time dependent operator for advancing the discretized conduction ODE
 */
class DynamicConductionOperator : public mfem::TimeDependentOperator {
protected:
  /**
   * @brief Finite Element space
   */
  std::shared_ptr<mfem::ParFiniteElementSpace> fespace_;

  /**
   * @brief Grid function for boundary condition projection
   */
  std::shared_ptr<mfem::ParGridFunction> state_gf_;

  /**
   * @brief Solver for the mass matrix
   */
  SystemSolver M_solver_;

  /**
   * @brief Solver for the T matrix
   */
  SystemSolver T_solver_;

  /**
   * @brief Pointer to the assembled M matrix
   */
  std::shared_ptr<mfem::HypreParMatrix> M_mat_;

  /**
   * @brief Pointer to the assembled K matrix
   */
  std::shared_ptr<mfem::HypreParMatrix> K_mat_;

  /**
   * @brief Pointer to the assembled T ( = M + dt K) matrix
   */
  std::unique_ptr<mfem::HypreParMatrix> T_mat_;

  /**
   * @brief Pointer to the eliminated T matrix
   */
  std::unique_ptr<mfem::HypreParMatrix> T_e_mat_;

  /**
   * @brief Assembled RHS vector
   */
  std::shared_ptr<mfem::Vector> rhs_;

  /**
   * @brief RHS vector including essential boundary elimination
   */
  std::shared_ptr<mfem::Vector> bc_rhs_;

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

public:
  /**
   * @brief Construct a new Dynamic Conduction Operator object
   *
   * @param[in] fespace The temperature field finite element space
   * @param[in] params The linear solver parameters
   * @param[in] ess_bdr The essential boundary condition objects
   */
  DynamicConductionOperator(std::shared_ptr<mfem::ParFiniteElementSpace> fespace,
                            const serac::LinearSolverParameters&         params,
                            std::vector<serac::BoundaryCondition>&       ess_bdr);

  /**
   * @brief Set the mass and stiffness matrices
   *
   * @param[in] M_mat The mass matrix
   * @param[in] K_mat The stiffness matrix
   */
  void setMatrices(std::shared_ptr<mfem::HypreParMatrix> M_mat, std::shared_ptr<mfem::HypreParMatrix> K_mat);

  /**
   * @brief Set the thermal flux load vector
   *
   * @param[in] The thermal flux (RHS vector)
   */
  void setLoadVector(std::shared_ptr<mfem::Vector> rhs);

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
};

}  // namespace serac

#endif

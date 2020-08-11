// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermal_solver.hpp
 *
 * @brief An object containing the solver for a thermal conduction PDE
 */

#ifndef CONDUCTION_SOLVER
#define CONDUCTION_SOLVER

#include "base_solver.hpp"
#include "mfem.hpp"
#include "thermal_operators.hpp"

namespace serac {

/**
 * @brief An object containing the solver for a thermal conduction PDE
 *
 * This is a generic linear thermal diffusion oeprator of the form
 *
 *    M du/dt = -kappa Ku + f
 *
 *  where M is a mass matrix, K is a stiffness matrix, and f is a
 *  thermal load vector.
 */
class ThermalSolver : public BaseSolver {
protected:
  /**
   * @brief The temperature finite element state
   */
  std::shared_ptr<serac::FiniteElementState> temperature_;

  /**
   * @brief Mass bilinear form object
   */
  std::unique_ptr<mfem::ParBilinearForm> M_form_;

  /**
   * @brief Stiffness bilinear form object
   */
  std::unique_ptr<mfem::ParBilinearForm> K_form_;

  /**
   * @brief Assembled mass matrix
   */
  std::shared_ptr<mfem::HypreParMatrix> M_mat_;

  /**
   * @brief Eliminated mass matrix
   */
  std::shared_ptr<mfem::HypreParMatrix> M_e_mat_;

  /**
   * @brief Assembled stiffness matrix
   */
  std::shared_ptr<mfem::HypreParMatrix> K_mat_;

  /**
   * @brief Eliminated stiffness matrix
   */
  std::shared_ptr<mfem::HypreParMatrix> K_e_mat_;

  /**
   * @brief Thermal load linear form
   */
  std::unique_ptr<mfem::ParLinearForm> l_form_;

  /**
   * @brief Assembled BC load vector
   */
  std::shared_ptr<mfem::HypreParVector> bc_rhs_;

  /**
   * @brief Assembled RHS vector
   */
  std::shared_ptr<mfem::HypreParVector> rhs_;

  /**
   * @brief Conduction coefficient
   */
  std::shared_ptr<mfem::Coefficient> kappa_;

  /**
   * @brief Body source coefficient
   */
  std::shared_ptr<mfem::Coefficient> source_;

  /**
   * @brief Time integration operator
   */
  std::unique_ptr<DynamicConductionOperator> dyn_oper_;

  /**
   * @brief Linear solver parameters
   */
  serac::LinearSolverParameters lin_params_;

  /**
   * @brief Solve the Quasi-static operator
   */
  void quasiStaticSolve();

public:
  /**
   * @brief Construct a new Thermal Solver object
   *
   * @param[in] order The order of the thermal field discretization
   * @param[in] pmesh The MFEM parallel mesh to solve the PDE on
   */
  ThermalSolver(int order, std::shared_ptr<mfem::ParMesh> pmesh);

  /**
   * @brief Set essential temperature boundary conditions (strongly enforced)
   *
   * @param[in] temp_bdr The boundary attributes on which to enforce a temperature
   * @param[in] temp_bdr_coef The prescribed boundary temperature
   */
  void setTemperatureBCs(const std::set<int>& temp_bdr, std::shared_ptr<mfem::Coefficient> temp_bdr_coef);

  /**
   * @brief Set flux boundary conditions (weakly enforced)
   *
   * @param[in] flux_bdr The boundary attributes on which to enforce a heat flux (weakly enforced)
   * @param[in] flux_bdr_coef The prescribed boundary heat flux
   */
  void setFluxBCs(const std::set<int>& flux_bdr, std::shared_ptr<mfem::Coefficient> flux_bdr_coef);

  /**
   * @brief Advance the timestep
   *
   * @param[in/out] dt The timestep to advance. For adaptive time integration methods, the actual timestep is returned.
   */
  void advanceTimestep(double& dt) override;

  /**
   * @brief Set the thermal conductivity
   *
   * @param[in] kappa The thermal conductivity
   */
  void setConductivity(std::shared_ptr<mfem::Coefficient> kappa);

  /**
   * @brief Set the temperature state vector from a coefficient
   *
   * @param[in] The temperature coefficient
   */
  void setTemperature(mfem::Coefficient& temp);

  /**
   * @brief Set the thermal body source from a coefficient
   *
   * @param[in] source The source function coefficient
   */
  void setSource(std::shared_ptr<mfem::Coefficient> source);

  /**
   * @brief Get the temperature state
   *
   * @return A pointer to the current temperature finite element state
   */
  std::shared_ptr<serac::FiniteElementState> temperature() { return temperature_; };

  /**
   * @brief Complete the initialization and allocation of the data structures.
   *
   * This must be called before StaticSolve() or AdvanceTimestep(). If allow_dynamic
   * = false, do not allocate the mass matrix or dynamic operator
   */
  void completeSetup() override;

  /**
   * @brief Set the linear solver parameters for both the M and K matrices
   *
   * @param[in] params The linear solver parameters
   */
  void setLinearSolverParameters(const serac::LinearSolverParameters& params);

  /**
   * @brief Destroy the Thermal Solver object
   */
  virtual ~ThermalSolver() = default;
};

}  // namespace serac

#endif

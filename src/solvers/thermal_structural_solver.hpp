// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermal_structural_solver.hpp
 *
 * @brief An object containing an operator-split thermal structural solver
 */

#ifndef THERMSTRUCT_SOLVER
#define THERMSTRUCT_SOLVER

#include "base_solver.hpp"
#include "mfem.hpp"
#include "nonlinear_solid_solver.hpp"
#include "thermal_solver.hpp"

namespace serac {

/**
 * @brief The operator-split thermal structural solver
 */
class ThermalStructuralSolver : public BaseSolver {
public:
  /**
   * @brief Construct a new Thermal Structural Solver object
   *
   * @param[in] order The order of the temperature and displacement discretizations
   * @param[in] mesh The parallel mesh object on which to solve
   */
  ThermalStructuralSolver(int order, std::shared_ptr<mfem::ParMesh> mesh);

  /**
   * @brief Set essential temperature boundary conditions (strongly enforced)
   *
   * @param[in] temp_bdr The attributes denotiving the fixed temperature boundary
   * @param[in] temp_bdr_coef The coefficient that contains the fixed temperature boundary values
   */
  void SetTemperatureBCs(const std::set<int>& temp_bdr, std::shared_ptr<mfem::Coefficient> temp_bdr_coef)
  {
    therm_solver_.setTemperatureBCs(temp_bdr, temp_bdr_coef);
  };

  /**
   * @brief Set flux boundary conditions (weakly enforced)
   *
   * @param[in] flux_bdr The boundary attributes on which to enforce a heat flux (weakly enforced)
   * @param[in] flux_bdr_coef The prescribed boundary heat flux
   */
  void SetFluxBCs(const std::set<int>& flux_bdr, std::shared_ptr<mfem::Coefficient> flux_bdr_coef)
  {
    therm_solver_.setFluxBCs(flux_bdr, flux_bdr_coef);
  };

  /**
   * @brief Set the thermal conductivity
   *
   * @param[in] kappa The thermal conductivity
   */
  void SetConductivity(std::unique_ptr<mfem::Coefficient>&& kappa) { therm_solver_.setConductivity(std::move(kappa)); };

  /**
   * @brief Set the temperature state vector from a coefficient
   *
   * @param[in] temp The temperature coefficient
   */
  void SetTemperature(mfem::Coefficient& temp) { therm_solver_.setTemperature(temp); };

  /**
   * @brief Set the thermal body source from a coefficient
   *
   * @param[in] source The source function coefficient
   */
  void SetSource(std::unique_ptr<mfem::Coefficient>&& source) { therm_solver_.setSource(std::move(source)); };

  /**
   * @brief Set the linear solver parameters for both the M and K matrices
   *
   * @param[in] params The linear solver parameters
   */
  void SetThermalSolverParameters(const serac::LinearSolverParameters& params)
  {
    therm_solver_.setLinearSolverParameters(params);
  };

  /**
   * @brief Set displacement boundary conditions
   *
   * @param[in] disp_bdr The set of boundary attributes to set the displacement on
   * @param[in] disp_bdr_coef The vector coefficient containing the set displacement values
   */
  void SetDisplacementBCs(const std::set<int>& disp_bdr, std::shared_ptr<mfem::VectorCoefficient> disp_bdr_coef)
  {
    solid_solver_.setDisplacementBCs(disp_bdr, disp_bdr_coef);
  };

  /**
   * @brief Set the displacement essential boundary conditions on a single component
   *
   * @param[in] disp_bdr The set of boundary attributes to set the displacement on
   * @param[in] disp_bdr_coef The vector coefficient containing the set displacement values
   * @param[in] component The component to set the displacment on
   */
  void SetDisplacementBCs(const std::set<int>& disp_bdr, std::shared_ptr<mfem::Coefficient> disp_bdr_coef,
                          const int component)
  {
    solid_solver_.setDisplacementBCs(disp_bdr, disp_bdr_coef, component);
  };

  /**
   * @brief Set the traction boundary conditions
   *
   * @param[in] trac_bdr The set of boundary attributes to apply a traction to
   * @param[in] trac_bdr_coef The vector valued traction coefficient
   * @param[in] component The component to apply the traction on
   */
  void SetTractionBCs(const std::set<int>& trac_bdr, std::shared_ptr<mfem::VectorCoefficient> trac_bdr_coef,
                      const int component = -1)
  {
    solid_solver_.setTractionBCs(trac_bdr, trac_bdr_coef, component);
  };

  /**
   * @brief Set the viscosity coefficient
   *
   * @param[in] visc_coef The abstract viscosity coefficient
   */
  void SetViscosity(std::unique_ptr<mfem::Coefficient>&& visc_coef)
  {
    solid_solver_.setViscosity(std::move(visc_coef));
  };

  /**
   * @brief Set the hyperelastic material parameters
   *
   * @param[in] mu Set the mu Lame parameter for the hyperelastic solid
   * @param[in] K Set the K Lame parameter for the hyperelastic solid
   */
  void SetHyperelasticMaterialParameters(double mu, double K)
  {
    solid_solver_.setHyperelasticMaterialParameters(mu, K);
  };

  /**
   * @brief Set the initial displacement value
   *
   * @param[in] disp_state The initial displacement state
   */
  void SetDisplacement(mfem::VectorCoefficient& disp_state) { solid_solver_.setDisplacement(disp_state); };

  /**
   * @brief Set the velocity state
   *
   * @param[in] velo_state The velocity state
   */
  void SetVelocity(mfem::VectorCoefficient& velo_state) { solid_solver_.setVelocity(velo_state); };

  /**
   * @brief Set the linear and nonlinear parameters for the solid solver object
   *
   * @param[in] lin_params The linear solver parameters
   * @param[in] nonlin_params The nonlinear solver parameters
   */
  void SetSolidSolverParameters(const serac::LinearSolverParameters&    lin_params,
                                const serac::NonlinearSolverParameters& nonlin_params)
  {
    solid_solver_.setSolverParameters(lin_params, nonlin_params);
  };

  /**
   * @brief Set the coupling scheme between the thermal and structural solvers
   *
   * Note that only operator split coupling is currently implemented.
   *
   * @param[in] coupling The coupling scheme
   */
  void SetCouplingScheme(serac::CouplingScheme coupling) { coupling_ = coupling; };

  /**
   * @brief Set the timestepping method for both the termal and structural solvers
   *
   * @param[in] timestepper The timestepping method
   */
  void setTimestepper(const serac::TimestepMethod timestepper) override;

  /**
   * @brief Complete the initialization and allocation of the data structures.
   *
   * This must be called before StaticSolve() or AdvanceTimestep(). If allow_dynamic
   * = false, do not allocate the mass matrix or dynamic operator.
   */
  void completeSetup() override;

  /**
   * @brief Get the temperature state
   *
   * @return A pointer to the current temperature finite element state
   */
  std::shared_ptr<serac::FiniteElementState> temperature() { return temperature_; };

  /**
   * @brief Get the displacement state
   *
   * @return The displacement state field
   */
  std::shared_ptr<serac::FiniteElementState> displacement() { return displacement_; };

  /**
   * @brief Get the velocity state
   *
   * @return The velocity state field
   */
  std::shared_ptr<serac::FiniteElementState> velocity() { return velocity_; };

  /**
   * @brief Advance the timestep
   *
   * @param[inout] dt The timestep to attempt. This will return the actual timestep for adaptive timestepping schemes
   */
  void advanceTimestep(double& dt) override;

  /**
   * @brief Destroy the Thermal Structural Solver object
   */
  virtual ~ThermalStructuralSolver() = default;

protected:
  /**
   * @brief The temperature finite element state
   */
  std::shared_ptr<serac::FiniteElementState> temperature_;

  /**
   * @brief The velocity finite element state
   */
  std::shared_ptr<serac::FiniteElementState> velocity_;

  /**
   * @brief The displacement finite element state
   */
  std::shared_ptr<serac::FiniteElementState> displacement_;

  /**
   * @brief The single physics thermal solver
   */
  ThermalSolver therm_solver_;

  /**
   * @brief The single physics nonlinear solid solver
   */
  NonlinearSolidSolver solid_solver_;

  /**
   * @brief The coupling strategy
   */
  serac::CouplingScheme coupling_;
};

}  // namespace serac

#endif

// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef THERMSTRUCT_SOLVER
#define THERMSTRUCT_SOLVER

#include "base_solver.hpp"
#include "mfem.hpp"
#include "nonlinear_solid_solver.hpp"
#include "thermal_solver.hpp"

/// The coupled thermal-structural solver object
class ThermalStructuralSolver : public BaseSolver {
 protected:
  /// The state variables
  std::shared_ptr<serac::FiniteElementState> temperature_;
  std::shared_ptr<serac::FiniteElementState> velocity_;
  std::shared_ptr<serac::FiniteElementState> displacement_;

  /// The thermal solver object
  ThermalSolver therm_solver_;

  /// The nonlinear solid solver object
  NonlinearSolidSolver solid_solver_;

  /// The coupling strategy
  serac::CouplingScheme coupling_;

 public:
  /// Constructor from order and parallel mesh
  ThermalStructuralSolver(int order, std::shared_ptr<mfem::ParMesh> pmesh);

  /// Set essential temperature boundary conditions (strongly enforced)
  void SetTemperatureBCs(const std::set<int> &temp_bdr, std::shared_ptr<mfem::Coefficient> temp_bdr_coef)
  {
    therm_solver_.SetTemperatureBCs(temp_bdr, temp_bdr_coef);
  };

  /// Set flux boundary conditions (weakly enforced)
  void SetFluxBCs(const std::set<int> &flux_bdr, std::shared_ptr<mfem::Coefficient> flux_bdr_coef)
  {
    therm_solver_.SetFluxBCs(flux_bdr, flux_bdr_coef);
  };

  /// Set the thermal conductivity coefficient
  void SetConductivity(std::shared_ptr<mfem::Coefficient> kappa) { therm_solver_.SetConductivity(kappa); };

  /// Set the temperature from a coefficient
  void SetTemperature(mfem::Coefficient &temp) { therm_solver_.SetTemperature(temp); };

  /// Set the body thermal source from a coefficient
  void SetSource(std::shared_ptr<mfem::Coefficient> source) { therm_solver_.SetSource(source); };

  /// Set the linear solver parameters for both the M and K operators
  void SetThermalSolverParameters(const serac::LinearSolverParameters &params)
  {
    therm_solver_.SetLinearSolverParameters(params);
  };

  /// Set the displacement essential boundary conditions
  void SetDisplacementBCs(const std::set<int> &disp_bdr, std::shared_ptr<mfem::VectorCoefficient> disp_bdr_coef)
  {
    solid_solver_.SetDisplacementBCs(disp_bdr, disp_bdr_coef);
  };

  /// Set the displacement essential boundary conditions on a single component
  void SetDisplacementBCs(const std::set<int> &disp_bdr, std::shared_ptr<mfem::Coefficient> disp_bdr_coef,
                          int component)
  {
    solid_solver_.SetDisplacementBCs(disp_bdr, disp_bdr_coef, component);
  };

  /// Set the traction boundary conditions
  void SetTractionBCs(const std::set<int> &trac_bdr, std::shared_ptr<mfem::VectorCoefficient> trac_bdr_coef,
                      int component = -1)
  {
    solid_solver_.SetTractionBCs(trac_bdr, trac_bdr_coef, component);
  };

  /// Set the viscosity coefficient
  void SetViscosity(std::shared_ptr<mfem::Coefficient> visc_coef) { solid_solver_.SetViscosity(visc_coef); };

  /// Set the hyperelastic material parameters
  void SetHyperelasticMaterialParameters(double mu, double K)
  {
    solid_solver_.SetHyperelasticMaterialParameters(mu, K);
  };

  /// Set the initial displacement state (guess)
  void SetDisplacement(mfem::VectorCoefficient &disp_state) { solid_solver_.SetDisplacement(disp_state); };

  /// Set the initial velocity state (guess)
  void SetVelocity(mfem::VectorCoefficient &velo_state) { solid_solver_.SetVelocity(velo_state); };

  /// Set the solid linear and nonlinear solver params
  void SetSolidSolverParameters(const serac::LinearSolverParameters &   lin_params,
                                const serac::NonlinearSolverParameters &nonlin_params)
  {
    solid_solver_.SetSolverParameters(lin_params, nonlin_params);
  };

  /// Set the coupling scheme
  void SetCouplingScheme(serac::CouplingScheme coupling) { coupling_ = coupling; };

  /// Overwrite the base default set timestepper method
  void SetTimestepper(serac::TimestepMethod timestepper);

  /** Complete the initialization and allocation of the data structures. This
   *  must be called before StaticSolve() or AdvanceTimestep(). If allow_dynamic
   * = false, do not allocate the mass matrix or dynamic operator */
  void CompleteSetup();

  /// Get the temperature state
  std::shared_ptr<serac::FiniteElementState> GetTemperature() { return temperature_; };

  /// Get the displacement state
  std::shared_ptr<serac::FiniteElementState> GetDisplacement() { return displacement_; };

  /// Get the velocity state
  std::shared_ptr<serac::FiniteElementState> GetVelocity() { return velocity_; };

  /// Advance the timestep
  void AdvanceTimestep(double &dt);

  /// Destructor
  virtual ~ThermalStructuralSolver() = default;
};

#endif

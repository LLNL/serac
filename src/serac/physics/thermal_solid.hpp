// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermal_solid.hpp
 *
 * @brief An object containing an operator-split thermal structural solver
 */

#pragma once

#include "mfem.hpp"

#include "serac/physics/base_physics.hpp"
#include "serac/physics/solid.hpp"
#include "serac/physics/thermal_conduction.hpp"

namespace serac {

/**
 * @brief The operator-split thermal structural solver
 */
class ThermalSolid : public BasePhysics {
public:
  /**
   * @brief Stores all information held in the input file that
   * is used to configure the thermal solid solver
   */
  struct InputOptions {
    /**
     * @brief Input file parameters specific to this class
     *
     * @param[in] container Inlet's Container that input files will be added to
     **/
    static void defineInputFileSchema(axom::inlet::Container& container);

    /**
     * @brief Solid mechanics input options
     */
    Solid::InputOptions solid_input;

    /**
     * @brief Thermal conduction input options
     *
     */
    ThermalConduction::InputOptions thermal_input;

    /**
     * @brief The isotropic coefficient of thermal expansion
     */
    std::optional<input::CoefficientInputOptions> coef_thermal_expansion;

    /**
     * @brief The reference temperature for thermal expansion
     */
    std::optional<input::CoefficientInputOptions> reference_temperature;
  };

  /**
   * @brief Construct a new Thermal Structural Solver object
   *
   * @param[in] order The order of the temperature and displacement discretizations
   * @param[in] therm_options The equation solver options for the conduction physics
   * @param[in] solid_options The equation solver options for the solid physics
   * @param[in] name A name for the physics module
   */
  ThermalSolid(int order, const ThermalConduction::SolverOptions& therm_options,
               const Solid::SolverOptions& solid_options, const std::string& name = "");

  /**
   * @brief Construct a new Thermal Solid object from input file options
   *
   * @param[in] thermal_solid_input The thermal solid physics module input file option struct
   * @param[in] name A name for the physics module
   */
  ThermalSolid(const ThermalSolid::InputOptions& thermal_solid_input, const std::string& name = "");

  /**
   * @brief Construct a new Thermal Solid object from input file options
   *
   * @param[in] thermal_input The thermal physics module input file option struct
   * @param[in] solid_input The solid physics module input file option struct
   * @param[in] name A name for the physics module
   */
  ThermalSolid(const ThermalConduction::InputOptions& thermal_input, const Solid::InputOptions& solid_input,
               const std::string& name = "");

  /**
   * @brief Set essential temperature boundary conditions (strongly enforced)
   *
   * @param[in] temp_bdr The attributes denotiving the fixed temperature boundary
   * @param[in] temp_bdr_coef The coefficient that contains the fixed temperature boundary values
   */
  void setTemperatureBCs(const std::set<int>& temp_bdr, std::shared_ptr<mfem::Coefficient> temp_bdr_coef)
  {
    therm_solver_.setTemperatureBCs(temp_bdr, temp_bdr_coef);
  };

  /**
   * @brief Set flux boundary conditions (weakly enforced)
   *
   * @param[in] flux_bdr The boundary attributes on which to enforce a heat flux (weakly enforced)
   * @param[in] flux_bdr_coef The prescribed boundary heat flux
   */
  void setFluxBCs(const std::set<int>& flux_bdr, std::shared_ptr<mfem::Coefficient> flux_bdr_coef)
  {
    therm_solver_.setFluxBCs(flux_bdr, flux_bdr_coef);
  };

  /**
   * @brief Set the thermal conductivity
   *
   * @param[in] kappa The thermal conductivity
   */
  void setConductivity(std::unique_ptr<mfem::Coefficient>&& kappa) { therm_solver_.setConductivity(std::move(kappa)); };

  /**
   * @brief Set the mass density
   *
   * @param[in] rho The mass density coefficient
   */
  void setMassDensity(std::unique_ptr<mfem::Coefficient>&& rho)
  {
    therm_solver_.setMassDensity(std::move(rho));
    solid_solver_.setMassDensity(std::move(rho));
  };

  /**
   * @brief Set the isotropic thermal expansion parameters
   *
   * @param coef_thermal_expansion The coefficient for thermal expansion
   * @param reference_temp The reference temperature
   */
  void setThermalExpansion(std::unique_ptr<mfem::Coefficient>&& coef_thermal_expansion,
                           std::unique_ptr<mfem::Coefficient>&& reference_temp)
  {
    solid_solver_.setThermalExpansion(std::move(coef_thermal_expansion), std::move(reference_temp),
                                      therm_solver_.temperature());
  }

  /**
   * @brief Set the specific heat capacity
   *
   * @param[in] cp The specific heat capacity
   */
  void setSpecificHeatCapacity(std::unique_ptr<mfem::Coefficient>&& cp)
  {
    therm_solver_.setSpecificHeatCapacity(std::move(cp));
  };

  /**
   * @brief Set the temperature state vector from a coefficient
   *
   * @param[in] temp The temperature coefficient
   */
  void setTemperature(mfem::Coefficient& temp) { therm_solver_.setTemperature(temp); };

  /**
   * @brief Set the thermal body source from a coefficient
   *
   * @param[in] source The source function coefficient
   */
  void setSource(std::unique_ptr<mfem::Coefficient>&& source) { therm_solver_.setSource(std::move(source)); };

  /**
   * @brief Set displacement boundary conditions
   *
   * @param[in] disp_bdr The set of boundary attributes to set the displacement on
   * @param[in] disp_bdr_coef The vector coefficient containing the set displacement values
   */
  void setDisplacementBCs(const std::set<int>& disp_bdr, std::shared_ptr<mfem::VectorCoefficient> disp_bdr_coef)
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
  void setDisplacementBCs(const std::set<int>& disp_bdr, std::shared_ptr<mfem::Coefficient> disp_bdr_coef,
                          const int component)
  {
    solid_solver_.setDisplacementBCs(disp_bdr, disp_bdr_coef, component);
  };

  /**
   * @brief Set the traction boundary conditions
   *
   * @param[in] trac_bdr The set of boundary attributes to apply a traction to
   * @param[in] trac_bdr_coef The vector valued traction coefficient
   * @param[in] compute_on_reference Flag to compute on the reference stress-free configuration vs. the deformed
   * configuration
   * @param[in] component The component to apply the traction on
   */
  void setTractionBCs(const std::set<int>& trac_bdr, std::shared_ptr<mfem::VectorCoefficient> trac_bdr_coef,
                      bool compute_on_reference, std::optional<int> component = {})
  {
    solid_solver_.setTractionBCs(trac_bdr, trac_bdr_coef, compute_on_reference, component);
  }

  /**
   * @brief Set the pressure boundary conditions
   *
   * @param[in] pres_bdr The set of boundary attributes to apply a pressure to
   * @param[in] pres_bdr_coef The scalar valued pressure coefficient
   * @param[in] compute_on_reference Flag to compute on the reference stress-free configuration vs. the deformed
   * configuration
   */
  void setPressureBCs(const std::set<int>& pres_bdr, std::shared_ptr<mfem::Coefficient> pres_bdr_coef,
                      bool compute_on_reference)
  {
    solid_solver_.setPressureBCs(pres_bdr, pres_bdr_coef, compute_on_reference);
  }

  /**
   * @brief Set the viscosity coefficient
   *
   * @param[in] visc_coef The abstract viscosity coefficient
   */
  void setViscosity(std::unique_ptr<mfem::Coefficient>&& visc_coef)
  {
    solid_solver_.setViscosity(std::move(visc_coef));
  };

  /**
   * @brief Set the material parameters
   *
   * @param[in] mu Set the shear modulus for the solid
   * @param[in] K Set the bulk modulus for the solid
   * @param[in] material_nonlin Flag to include material nonlinearities (linear elastic vs. neo-Hookean model)
   */
  void setSolidMaterialParameters(std::unique_ptr<mfem::Coefficient>&& mu, std::unique_ptr<mfem::Coefficient>&& K,
                                  bool material_nonlin = true)
  {
    solid_solver_.setMaterialParameters(std::move(mu), std::move(K), material_nonlin);
  };

  /**
   * @brief Set the initial displacement value
   *
   * @param[in] disp_state The initial displacement state
   */
  void setDisplacement(mfem::VectorCoefficient& disp_state) { solid_solver_.setDisplacement(disp_state); };

  /**
   * @brief Set the velocity state
   *
   * @param[in] velo_state The velocity state
   */
  void setVelocity(mfem::VectorCoefficient& velo_state) { solid_solver_.setVelocity(velo_state); };

  /**
   * @brief Set the coupling scheme between the thermal and structural solvers
   *
   * Note that only operator split coupling is currently implemented.
   *
   * @param[in] coupling The coupling scheme
   */
  void setCouplingScheme(serac::CouplingScheme coupling) { coupling_ = coupling; };

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
   * @return A reference to the current temperature finite element state
   */
  const serac::FiniteElementState& temperature() { return temperature_; };

  /**
   * @brief Get the displacement state
   *
   * @return The displacement state field
   */
  const serac::FiniteElementState& displacement() { return displacement_; };

  /**
   * @brief Get the velocity state
   *
   * @return The velocity state field
   */
  const serac::FiniteElementState& velocity() { return velocity_; };

  /**
   * @brief Advance the timestep
   *
   * @param[inout] dt The timestep to attempt. This will return the actual timestep for adaptive timestepping schemes
   */
  void advanceTimestep(double& dt) override;

  /**
   * @brief Destroy the Thermal Structural Solver object
   */
  virtual ~ThermalSolid() = default;

protected:
  /**
   * @brief The single physics thermal solver
   */
  ThermalConduction therm_solver_;

  /**
   * @brief The single physics nonlinear solid solver
   */
  Solid solid_solver_;

  /**
   * @brief The temperature finite element state
   */
  const serac::FiniteElementState& temperature_;

  /**
   * @brief The velocity finite element state
   */
  const serac::FiniteElementState& velocity_;

  /**
   * @brief The displacement finite element state
   */
  const serac::FiniteElementState& displacement_;

  /**
   * @brief The coupling strategy
   */
  serac::CouplingScheme coupling_;
};

}  // namespace serac

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by inlet
 */
template <>
struct FromInlet<serac::ThermalSolid::InputOptions> {
  /// @brief Returns created object from Inlet container
  serac::ThermalSolid::InputOptions operator()(const axom::inlet::Container& base);
};

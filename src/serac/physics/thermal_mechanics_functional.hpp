// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermal_mechanics_functional.hpp
 *
 * @brief An object containing an operator-split thermal structural solver
 */

#pragma once

#include "mfem.hpp"

#include "serac/physics/base_physics.hpp"
#include "serac/physics/solid_functional.hpp"
#include "serac/physics/thermal_conduction_functional.hpp"
#include "serac/physics/materials/thermal_functional_material.hpp"
#include "serac/physics/materials/solid_functional_material.hpp"

namespace serac {

/**
 * @brief The operator-split thermal-structural solver
 *
 * Uses Functional to compute action of operators
 */
template <int order, int dim, typename... parameter_space>
class ThermalMechanicsFunctional : public BasePhysics {
public:
  /**
   * @brief Construct a new coupled Thermal-Solid Functional object
   *
   * @param thermal_options The options for the linear, nonlinear, and ODE solves of the thermal operator
   * @param solid_options The options for the linear, nonlinear, and ODE solves of the thermal operator
   * @param geom_nonlin Flag to include geometric nonlinearities
   * @param name An optional name for the physics module instance
   */
  ThermalMechanicsFunctional(const SolverOptions& thermal_options, const SolverOptions& solid_options,
                             GeometricNonlinearities geom_nonlin = GeometricNonlinearities::On,
                             const std::string&      name        = "")
      : BasePhysics(3, order, name),
        thermal_functional_(thermal_options, name + "thermal"),
        solid_functional_(solid_options, geom_nonlin, name + "mechanical")
  {
    SLIC_ERROR_ROOT_IF(mesh_.Dimension() != dim,
                       axom::fmt::format("Compile time dimension and runtime mesh dimension mismatch"));

    state_.push_back(thermal_functional_.temperature());
    state_.push_back(solid_functional_.velocity());
    state_.push_back(solid_functional_.displacement());

    thermal_functional_.setParameter(solid_functional_.displacement(), 0);
    solid_functional_.setParameter(thermal_functional_.temperature(), 0);

    coupling_ = serac::CouplingScheme::OperatorSplit;
  }

  /**
   * @brief Complete the initialization and allocation of the data structures.
   *
   * @note This must be called before AdvanceTimestep().
   */
  void completeSetup() override
  {
    SLIC_ERROR_ROOT_IF(coupling_ != serac::CouplingScheme::OperatorSplit,
                       "Only operator split is currently implemented in the thermal structural solver.");

    thermal_functional_.completeSetup();
    solid_functional_.completeSetup();
  }

  /**
   * @brief register the provided FiniteElementState object as the source of values for parameter `i`
   *
   * @param parameter_state the values to use for the specified parameter
   * @param i the index of the parameter
   */
  void setParameter(const FiniteElementState& parameter_state, size_t i)
  {
    thermal_functional_.setParameter(parameter_state, i + 1);  // offset for displacement field
    solid_functional_.setParameter(parameter_state, i + 1);    // offset for temperature field
  }

  /**
   * @brief Advance the timestep
   *
   * @param[inout] dt The timestep to attempt. This will return the actual timestep for adaptive timestepping
   * schemes
   * @pre completeSetup() must be called prior to this call
   */
  void advanceTimestep(double& dt) override
  {
    if (coupling_ == serac::CouplingScheme::OperatorSplit) {
      double initial_dt = dt;
      thermal_functional_.advanceTimestep(dt);
      solid_functional_.advanceTimestep(dt);
      SLIC_ERROR_ROOT_IF(std::abs(dt - initial_dt) > 1.0e-6,
                         "Operator split coupled solvers cannot adaptively change the timestep");
    } else {
      SLIC_ERROR_ROOT("Only operator split coupling is currently implemented");
    }

    cycle_ += 1;
  }

  /**
   * @brief Create a shared ptr to a quadrature data buffer for the given material type
   *
   * @tparam T the type to be created at each quadrature point
   * @param initial_state the value to be broadcast to each quadrature point
   * @return std::shared_ptr< QuadratureData<T> >
   */
  template <typename T>
  std::shared_ptr<QuadratureData<T>> createQuadratureDataBuffer(T initial_state)
  {
    return solid_functional_.createQuadratureDataBuffer(initial_state);
  }

  /**
   * @brief This is an adaptor class that makes a thermomechanical material usable by
   * the thermal module, by discarding the solid-mechanics-specific information
   *
   * @tparam ThermalMechanicalMaterial the material model being wrapped
   */
  template <typename ThermalMechanicalMaterial>
  struct ThermalMaterialInterface {
    using State = typename ThermalMechanicalMaterial::State;  ///< internal variables for the wrapped material model

    const ThermalMechanicalMaterial mat;  ///< the wrapped material model

    /// constructor
    ThermalMaterialInterface(const ThermalMechanicalMaterial& m) : mat(m)
    {
      // empty
    }

    /**
     * @brief glue code to evaluate a thermomechanical material and extract the thermal outputs
     *
     * @tparam T1 the type of the spatial coordinate values
     * @tparam T2 the type of the temperature value
     * @tparam T3 the type of the temperature gradient values
     * @tparam T4 the type of the displacement gradient values
     * @tparam param_types the types of user-specified parameters
     * @param temperature the temperature at this quadrature point
     * @param temperature_gradient the gradient w.r.t. physical coordinates of the temperature
     * @param displacement the value and gradient w.r.t. physical coordinates of the displacement
     * @param parameters values and derivatives of any additional user-specified parameters
     */
    template <typename T1, typename T2, typename T3, typename T4, typename... param_types>
    SERAC_HOST_DEVICE auto operator()(const T1& /* x */, const T2& temperature, const T3& temperature_gradient,
                                      const T4& displacement, param_types... parameters) const
    {
      // BT: this will not update the state correctly. I just want to get the code compiling before plumbing the state
      // variables.
      State state{};

      auto [u, du_dX]     = displacement;
      auto [T, c, s0, q0] = mat(state, du_dX, temperature, temperature_gradient, parameters...);
      // density * specific_heat = c
      return Thermal::MaterialResponse{mat.density, c, q0};
    }
  };

  /**
   * @brief This is an adaptor class that makes a thermomechanical material usable by
   * the solid mechanics module, by discarding the thermal-specific information
   *
   * @tparam ThermalMechanicalMaterial the material model being wrapped
   */
  template <typename ThermalMechanicalMaterial>
  struct MechanicalMaterialInterface {
    using State = typename ThermalMechanicalMaterial::State;  ///< internal variables for the wrapped material model

    const ThermalMechanicalMaterial mat;  ///< the wrapped material model

    const double density;  ///< mass density

    /// constructor
    MechanicalMaterialInterface(const ThermalMechanicalMaterial& m) : mat(m), density(m.density)
    {
      // empty
    }

    /**
     * @brief glue code to evaluate a thermomechanical material and extract the stress
     *
     * @tparam T1 the type of the displacement gradient values
     * @tparam T2 the type of the temperature value
     * @tparam param_types the types of user-specified parameters
     * @param state any internal variables needed to evaluate this material
     * @param displacement_gradient the gradient w.r.t. physical coordinates of the displacement
     * @param temperature the temperature at this quadrature point
     * @param parameters values and derivatives of any additional user-specified parameters
     */
    template <typename T1, typename T2, typename... param_types>
    SERAC_HOST_DEVICE auto operator()(State& state, const T1& displacement_gradient, const T2& temperature,
                                      param_types... parameters) const
    {
      auto [theta, dtheta_dX] = temperature;
      auto [T, c, s0, q0]     = mat(state, displacement_gradient, theta, dtheta_dX, parameters...);
      return T;
    }
  };

  /**
   * @brief Set the material response for the physics module
   *
   * @tparam MaterialType The type of material model
   * @tparam StateType The type that contains the internal variables for MaterialType
   * @param material A material that provides a function to evaluate stress, heat flux
   * @param qdata the buffer of material internal variables at each quadrature point
   *
   * @pre MaterialType must have a public member variable `density`
   * @pre MaterialType must define operator() that returns the Kirchoff stress, heat flux and heat source terms
   */
  template <int... active_parameters, typename MaterialType, typename StateType>
  void setMaterial(DependsOn<active_parameters...>, MaterialType material,
                   std::shared_ptr<QuadratureData<StateType>> qdata)
  {
    // note: these parameter indices are offset by 1 since, internally, this module uses the first parameter
    // to communicate the temperature and displacement field information to the other physics module
    //
    thermal_functional_.setMaterial(DependsOn<0, active_parameters + 1 ...>{},
                                    ThermalMaterialInterface<MaterialType>{material});
    solid_functional_.setMaterial(DependsOn<0, active_parameters + 1 ...>{},
                                  MechanicalMaterialInterface<MaterialType>{material}, qdata);
  }

  /// @overload
  template <typename MaterialType, typename StateType = Empty>
  void setMaterial(MaterialType material, std::shared_ptr<QuadratureData<StateType>> qdata = EmptyQData)
  {
    setMaterial(DependsOn<>{}, material, qdata);
  }

  /**
   * @brief Set essential temperature boundary conditions (strongly enforced)
   *
   * @param[in] temperature_attributes The boundary attributes on which to enforce a temperature
   * @param[in] prescribed_value The prescribed boundary temperature function
   */
  void setTemperatureBCs(const std::set<int>&                                   temperature_attributes,
                         std::function<double(const mfem::Vector& x, double t)> prescribed_value)
  {
    thermal_functional_.setTemperatureBCs(temperature_attributes, prescribed_value);
  }

  /**
   * @brief Set essential displacement boundary conditions (strongly enforced)
   *
   * @param[in] displacement_attributes The boundary attributes on which to enforce a displacement
   * @param[in] prescribed_value The prescribed boundary displacement function
   */
  void setDisplacementBCs(const std::set<int>&                                           displacement_attributes,
                          std::function<void(const mfem::Vector& x, mfem::Vector& disp)> prescribed_value)
  {
    solid_functional_.setDisplacementBCs(displacement_attributes, prescribed_value);
  }

  /**
   * @brief Set the thermal flux boundary condition
   *
   * @tparam FluxType The type of the flux function
   * @param flux_function A function describing the thermal flux applied to a boundary
   *
   * @pre FluxType must have the operator (x, normal, temperature) to return the thermal flux value
   */
  template <typename FluxType>
  void setHeatFluxBCs(FluxType flux_function)
  {
    thermal_functional_.setFluxBCs(flux_function);
  }

  /**
   * @brief Set the underlying finite element state to a prescribed displacement
   *
   * @param displacement The function describing the displacement field
   */
  void setDisplacement(std::function<void(const mfem::Vector& x, mfem::Vector& u)> displacement)
  {
    solid_functional_.setDisplacement(displacement);
  }

  /**
   * @brief Set the underlying finite element state to a prescribed temperature
   *
   * @param temperature The function describing the temperature field
   */
  void setTemperature(std::function<double(const mfem::Vector& x, double t)> temperature)
  {
    thermal_functional_.setTemperature(temperature);
  }

  /**
   * @brief Set the body forcefunction
   *
   * @tparam BodyForceType The type of the body force load
   * @param body_force_function A source function for a prescribed body load
   *
   * @pre BodyForceType must have the operator (x, time) defined as the body force
   */
  template <typename BodyForceType>
  void addBodyForce(BodyForceType body_force_function)
  {
    solid_functional_.addBodyForce(body_force_function);
  }

  /**
   * @brief Set the thermal source function
   *
   * @tparam SourceType The type of the source function
   * @param source_function A source function for a prescribed thermal load
   *
   * @pre SourceType must have the operator (x, time, temperature, d temperature_dx) defined as the thermal source
   */
  template <typename HeatSourceType>
  void addHeatSource(HeatSourceType source_function)
  {
    thermal_functional_.setSource(source_function);
  }

  /**
   * @brief Get the displacement state
   *
   * @return A reference to the current displacement finite element state
   */
  const serac::FiniteElementState& displacement() const { return solid_functional_.displacement(); };

  /**
   * @brief Get the temperature state
   *
   * @return A reference to the current temperature finite element state
   */
  const serac::FiniteElementState& temperature() const { return thermal_functional_.temperature(); };

protected:
  /// @brief The coupling strategy
  serac::CouplingScheme coupling_;

  using displacement_field = H1<order, dim>;  ///< the function space for the displacement field
  using temperature_field  = H1<order>;       ///< the function space for the temperature field

  /// Submodule to compute the thermal conduction physics
  ThermalConductionFunctional<order, dim, Parameters<displacement_field, parameter_space...>> thermal_functional_;

  /// Submodule to compute the mechanics
  SolidFunctional<order, dim, Parameters<temperature_field, parameter_space...>> solid_functional_;
};

}  // namespace serac

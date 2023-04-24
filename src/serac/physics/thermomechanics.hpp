// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermomechanics.hpp
 *
 * @brief An object containing an operator-split thermal structural solver
 */

#pragma once

#include "mfem.hpp"

#include "serac/physics/base_physics.hpp"
#include "serac/physics/solid_mechanics.hpp"
#include "serac/physics/heat_transfer.hpp"
#include "serac/physics/materials/thermal_material.hpp"
#include "serac/physics/materials/solid_material.hpp"

namespace serac {

/**
 * @brief The operator-split thermal-structural solver
 *
 * Uses Functional to compute action of operators
 */
template <int order, int dim, typename... parameter_space>
class Thermomechanics : public BasePhysics {
public:
  /**
   * @brief Construct a new coupled Thermal-SolidMechanics Functional object
   *
   * @param thermal_solver The nonlinear equation solver for the heat conduction equations
   * @param thermal_timestepping The timestepping options for the thermal solver
   * @param solid_solver The nonlinear equation solver for the solid mechanics equations
   * @param solid_timestepping The timestepping options for the solid solver
   * @param geom_nonlin Flag to include geometric nonlinearities
   * @param name An optional name for the physics module instance
   * @param pmesh The mesh to conduct the simulation on, if different than the default mesh
   */
  Thermomechanics(std::unique_ptr<mfem_ext::EquationSolver> thermal_solver, TimesteppingOptions thermal_timestepping,
                  std::unique_ptr<mfem_ext::EquationSolver> solid_solver, TimesteppingOptions solid_timestepping,
                  GeometricNonlinearities geom_nonlin = GeometricNonlinearities::On, const std::string& name = "",
                  mfem::ParMesh* pmesh = nullptr)
      : BasePhysics(3, order, name, pmesh),
        thermal_(std::move(thermal_solver), thermal_timestepping, name + "thermal", pmesh),
        solid_(std::move(solid_solver), solid_timestepping, geom_nonlin, name + "mechanical", pmesh)
  {
    SLIC_ERROR_ROOT_IF(mesh_.Dimension() != dim,
                       axom::fmt::format("Compile time dimension and runtime mesh dimension mismatch"));

    states_.push_back(&thermal_.temperature());
    states_.push_back(&solid_.velocity());
    states_.push_back(&solid_.displacement());

    thermal_.setParameter(0, solid_.displacement());
    solid_.setParameter(0, thermal_.temperature());
  }

  /**
   * @brief Complete the initialization and allocation of the data structures.
   *
   * @note This must be called before AdvanceTimestep().
   */
  void completeSetup() override
  {
    thermal_.completeSetup();
    solid_.completeSetup();
  }

  /**
   * @brief register the provided FiniteElementState object as the source of values for parameter `i`
   *
   * @param parameter_state the values to use for the specified parameter
   * @param i the index of the parameter
   */
  void setParameter(const FiniteElementState& parameter_state, size_t i)
  {
    thermal_.setParameter(parameter_state, i + 1);  // offset for displacement field
    solid_.setParameter(parameter_state, i + 1);    // offset for temperature field
  }

  /**
   * @brief Accessor for getting named finite element state fields from the physics modules
   *
   * @param state_name The name of the Finite Element State to retrieve
   * @return The named Finite Element State
   */
  const FiniteElementState& state(const std::string& state_name) override
  {
    if (state_name == "displacement") {
      return solid_.displacement();
    } else if (state_name == "velocity") {
      return solid_.velocity();
    } else if (state_name == "temperature") {
      return thermal_.temperature();
    }

    SLIC_ERROR_ROOT(axom::fmt::format("State '{}' requestion from solid mechanics module '{}', but it doesn't exist",
                                      state_name, name_));
    return solid_.displacement();
  }

  /**
   * @brief Get a vector of the finite element state solution variable names
   *
   * @return The solution variable names
   */
  virtual std::vector<std::string> stateNames() override
  {
    return std::vector<std::string>{{"displacement"}, {"velocity"}, {"temperature"}};
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
    double initial_dt = dt;
    thermal_.advanceTimestep(dt);
    solid_.advanceTimestep(dt);
    SLIC_ERROR_ROOT_IF(std::abs(dt - initial_dt) > 1.0e-6,
                       "Operator split coupled solvers cannot adaptively change the timestep");

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
    return solid_.createQuadratureDataBuffer(initial_state);
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

      auto [u, du_dX]                 = displacement;
      auto [T, heat_capacity, s0, q0] = mat(state, du_dX, temperature, temperature_gradient, parameters...);

      return serac::tuple{heat_capacity, q0};
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
      auto [theta, dtheta_dX]         = temperature;
      auto [T, heat_capacity, s0, q0] = mat(state, displacement_gradient, theta, dtheta_dX, parameters...);
      return T;
    }
  };

  /**
   * @brief Set the thermomechanical material response
   *
   * @tparam MaterialType The thermomechanical material type
   * @tparam StateType The type that contains the internal variables for MaterialType
   * @param material A material that provides a function to evaluate stress, heat flux, density, and heat capacity
   * @param qdata the buffer of material internal variables at each quadrature point
   *
   * @pre material must be a object that can be called with the following arguments:
   *    1. `MaterialType::State & state` an mutable reference to the internal variables for this quadrature point
   *    2. `tensor<T,dim,dim> du_dx` the displacement gradient at this quadrature point
   *    3. `T temperature` the current temperature at the quadrature point
   *    4. `tensor<T,dim>` the spatial gradient of the temperature at the quadrature point
   *    5. `tuple{value, derivative}`, a tuple of values and derivatives for each parameter field
   *            specified in the `DependsOn<...>` argument.
   *
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   * @pre MaterialType must return a serac::tuple of Cauchy stress, volumetric heat capacity, internal heat source, and
   * thermal flux when operator() is called with the arguments listed above.
   */
  template <int... active_parameters, typename MaterialType, typename StateType>
  void setMaterial(DependsOn<active_parameters...>, MaterialType material,
                   std::shared_ptr<QuadratureData<StateType>> qdata)
  {
    // note: these parameter indices are offset by 1 since, internally, this module uses the first parameter
    // to communicate the temperature and displacement field information to the other physics module
    //
    thermal_.setMaterial(DependsOn<0, active_parameters + 1 ...>{}, ThermalMaterialInterface<MaterialType>{material});
    solid_.setMaterial(DependsOn<0, active_parameters + 1 ...>{}, MechanicalMaterialInterface<MaterialType>{material},
                       qdata);
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
    thermal_.setTemperatureBCs(temperature_attributes, prescribed_value);
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
    solid_.setDisplacementBCs(displacement_attributes, prescribed_value);
  }

  /**
   * @brief Set the thermal flux boundary condition
   *
   * @tparam FluxType The type of the thermal flux object
   * @param flux_function A function describing the flux applied to a boundary
   *
   * @pre FluxType must be a object that can be called with the following arguments:
   *    1. `tensor<T,dim> x` the spatial coordinates for the quadrature point
   *    2. `tensor<T,dim> n` the outward-facing unit normal for the quadrature point
   *    3. `double t` the time (note: time will be handled differently in the future)
   *    4. `T temperature` the current temperature at the quadrature point
   *    4. `tuple{value, derivative}`, a variadic list of tuples (each with a values and derivative),
   *            one tuple for each of the trial spaces specified in the `DependsOn<...>` argument.
   *
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   * @note: until mfem::GetFaceGeometricFactors implements their JACOBIANS option,
   * (or we implement a replacement kernel ourselves) we are not able to compute
   * shape sensitivities for boundary integrals.
   */
  template <typename FluxType>
  void setHeatFluxBCs(FluxType flux_function)
  {
    thermal_.setFluxBCs(flux_function);
  }

  /**
   * @brief Set the underlying finite element state to a prescribed displacement
   *
   * @param displacement The function describing the displacement field
   */
  void setDisplacement(std::function<void(const mfem::Vector& x, mfem::Vector& u)> displacement)
  {
    solid_.setDisplacement(displacement);
  }

  /**
   * @brief Set the underlying finite element state to a prescribed temperature
   *
   * @param temperature The function describing the temperature field
   */
  void setTemperature(std::function<double(const mfem::Vector& x, double t)> temperature)
  {
    thermal_.setTemperature(temperature);
  }

  /**
   * @brief Set the body forcefunction
   *
   * @tparam BodyForceType The type of the body force load
   * @pre body_force_function must be a object that can be called with the following arguments:
   *    1. `tensor<T,dim> x` the spatial coordinates for the quadrature point
   *    2. `double t` the time (note: time will be handled differently in the future)
   *    3. `tuple{value, derivative}`, a variadic list of tuples (each with a values and derivative),
   *            one tuple for each of the trial spaces specified in the `DependsOn<...>` argument.
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   */
  template <typename BodyForceType>
  void addBodyForce(BodyForceType body_force_function)
  {
    solid_.addBodyForce(body_force_function);
  }

  /**
   * @brief Set the thermal source function
   *
   * @tparam HeatSourceType The type of the source function
   * @param source_function A source function for a prescribed thermal load
   *
   * @pre source_function must be a object that can be called with the following arguments:
   *    1. `tensor<T,dim> x` the spatial coordinates for the quadrature point
   *    2. `double t` the time (note: time will be handled differently in the future)
   *    3. `T temperature` the current temperature at the quadrature point
   *    4. `tensor<T,dim>` the spatial gradient of the temperature at the quadrature point
   *    5. `tuple{value, derivative}`, a variadic list of tuples (each with a values and derivative),
   *            one tuple for each of the trial spaces specified in the `DependsOn<...>` argument.
   *
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   */
  template <typename HeatSourceType>
  void addHeatSource(HeatSourceType source_function)
  {
    thermal_.setSource(source_function);
  }

  /**
   * @brief Get the displacement state
   *
   * @return A reference to the current displacement finite element state
   */
  const serac::FiniteElementState& displacement() const { return solid_.displacement(); };

  /**
   * @brief Get the temperature state
   *
   * @return A reference to the current temperature finite element state
   */
  const serac::FiniteElementState& temperature() const { return thermal_.temperature(); };

protected:
  using displacement_field = H1<order, dim>;  ///< the function space for the displacement field
  using temperature_field  = H1<order>;       ///< the function space for the temperature field

  /// Submodule to compute the thermal conduction physics
  HeatTransfer<order, dim, Parameters<displacement_field, parameter_space...>> thermal_;

  /// Submodule to compute the mechanics
  SolidMechanics<order, dim, Parameters<temperature_field, parameter_space...>> solid_;
};

}  // namespace serac

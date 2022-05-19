// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermal_solid_functional.hpp
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
template <int order, int dim>
class ThermalMechanicsFunctional : public BasePhysics {
public:
  /**
   * @brief Construct a new coupled Thermal-Mechanics Functional object
   *
   * @param thermal_options The options for the linear, nonlinear, and ODE solves of the thermal operator
   * @param solid_options The options for the linear, nonlinear, and ODE solves of the thermal operator
   * @param geom_nonlin Flag to include geometric nonlinearities
   * @param keep_deformation Flag to keep the deformation in the underlying mesh post-destruction
   * @param name An optional name for the physics module instance
   */
  ThermalMechanicsFunctional(const typename Thermal::SolverOptions&    thermal_options,
                             const typename solid_util::SolverOptions& solid_options,
                             GeometricNonlinearities                   geom_nonlin = GeometricNonlinearities::On,
                             FinalMeshOption keep_deformation = FinalMeshOption::Deformed, const std::string& name = "")
      : BasePhysics(3, order),
        temperature_(
            StateManager::newState(FiniteElementState::Options{.order      = order,
                                                               .vector_dim = 1,
                                                               .ordering   = mfem::Ordering::byNODES,
                                                               .name       = detail::addPrefix(name, "temperature")})),
        velocity_(StateManager::newState(FiniteElementState::Options{
            .order = order, .vector_dim = mesh_.Dimension(), .name = detail::addPrefix(name, "velocity")})),
        displacement_(StateManager::newState(FiniteElementState::Options{
            .order = order, .vector_dim = mesh_.Dimension(), .name = detail::addPrefix(name, "displacement")})),
        thermal_functional_(thermal_options, name + "thermal", {displacement_}),
        solid_functional_(solid_options, geom_nonlin, keep_deformation, name + "mechanical", {temperature_})
  {
    SLIC_ERROR_ROOT_IF(mesh_.Dimension() != dim,
                       axom::fmt::format("Compile time dimension and runtime mesh dimension mismatch"));

    state_.push_back(thermal_functional_.temperature());
    state_.push_back(solid_functional_.velocity());
    state_.push_back(solid_functional_.displacement());

    coupling_ = serac::CouplingScheme::OperatorSplit;
  }

  void completeSetup() override
  {
    SLIC_ERROR_ROOT_IF(coupling_ != serac::CouplingScheme::OperatorSplit,
                       "Only operator split is currently implemented in the thermal structural solver.");

    thermal_functional_.completeSetup();
    solid_functional_.completeSetup();
  }

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

  template <typename ThermalMechanicalMaterial>
  struct ThermalMaterialInterface {
    const ThermalMechanicalMaterial mat;

    ThermalMaterialInterface(const ThermalMechanicalMaterial& m):
        mat(m)
    {
      // empty
    }

    static constexpr int numParameters() { return 1; }

    template <typename T1, typename T2, typename T3, typename T4>
    SERAC_HOST_DEVICE auto operator()(const T1& /* x */, const T2& temperature,
                                      const T3& temperature_gradient, const T4& displacement) const
    {
      typename ThermalMechanicalMaterial::State state{};
      const double dt = 1.0;
      auto [u, du_dX] = displacement;
      auto du_dX_old = tensor<double, 3, 3>{};
      double temperature_old = 1.0;
      auto [P, c, s0, q0] = mat.calculateConstitutiveOutputs(du_dX, temperature, temperature_gradient,
                                                             state, du_dX_old, temperature_old, dt);
      // density * specific_heat = c
      const double density = 1.0;
      return Thermal::MaterialResponse{density, c, q0};
    }
  };

  template <typename ThermalMechanicalMaterial>
  struct MechanicalMaterialInterface {
    const ThermalMechanicalMaterial mat;

    MechanicalMaterialInterface(const ThermalMechanicalMaterial& m):
        mat(m)
    {
      // empty
    }

    static constexpr int numParameters() { return 1; }

    template <typename T1, typename T2, typename T3, typename T4>
    SERAC_HOST_DEVICE auto operator()(const T1& /* x */, const T2& /* displacement */,
                                      const T3& displacement_gradient, const T4& temperature) const
    {
      typename ThermalMechanicalMaterial::State state{};
      const double dt = 1.0;
      auto [theta, dtheta_dX] = temperature;
      double temperature_old = 1.0;
      auto displacement_gradient_old = tensor<double, 3, 3>{};
      auto [P, c, s0, q0] = mat.calculateConstitutiveOutputs(displacement_gradient, theta, dtheta_dX,
                                                             state, displacement_gradient_old, temperature_old, dt);
      const double density = 1.0;
      auto F = displacement_gradient + Identity<3>();
      auto stress = dot(P, transpose(F));
      return solid_util::MaterialResponse{density, stress};
    }
  };
  
  template <typename MaterialType>
  void setMaterial(MaterialType material)
  {
    thermal_functional_.setMaterial(ThermalMaterialInterface<MaterialType>{material});
    solid_functional_.setMaterial(MechanicalMaterialInterface<MaterialType>{material});
  }

  /**
   * @brief Set essential temperature boundary conditions (strongly enforced)
   *
   * @param[in] temperature_attributes The boundary attributes on which to enforce a temperature
   * @param[in] prescribed_value The prescribed boundary temperature function
   */
  void setTemperatureBCs(const std::set<int>& temperature_attributes,
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
  void setDisplacementBCs(const std::set<int>& displacement_attributes,
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
   * @brief Get the temperature state
   *
   * @return A reference to the current temperature finite element state
   */
  const serac::FiniteElementState& temperature() const { return temperature_; };

  /// @overload
  //serac::FiniteElementState& temperature() { return temperature_; };

  
protected:
  /// The temperature finite element state
  FiniteElementState temperature_;

  /// The velocity finite element state
  FiniteElementState velocity_;

  /// The displacement finite element state
  FiniteElementState displacement_;

  /**
   * @brief The coupling strategy
   */
  serac::CouplingScheme coupling_;

  /// A thermal functional module
  ThermalConductionFunctional<order, dim, displacement_field> thermal_functional_;

  /// A solid functional module
  SolidFunctional<order, dim, temperature_field> solid_functional_;
};

}  // namespace serac

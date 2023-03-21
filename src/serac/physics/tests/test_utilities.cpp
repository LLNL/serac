// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_utilities.hpp"

#include "axom/core.hpp"
#include <gtest/gtest.h>

#include "serac/infrastructure/input.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/solid_legacy.hpp"
#include "serac/physics/thermal_conduction_legacy.hpp"
#include "serac/physics/thermal_solid_legacy.hpp"
#include "serac/physics/state/state_manager.hpp"

namespace serac {

namespace test_utils {

/**
 * @brief Defines elements of a test schema that are not specific to a particular
 * physics module
 * @param[inout] inlet The top-level Inlet object to define the schema on
 * @note This function should be called *after* defining any physics-specific parts of
 * the input file
 */
void defineCommonTestSchema(axom::inlet::Inlet& inlet)
{
  // Simulation time parameters
  inlet.addDouble("dt", "Time step.");
  inlet.addDouble("t_final", "Stopping point");

  // Comparison parameter
  inlet.addDouble("epsilon", "Threshold to be used in the comparison");

  auto& mesh_table = inlet.addStruct("main_mesh", "The main mesh for the problem");
  serac::mesh::InputOptions::defineInputFileSchema(mesh_table);

  // Verify input file
  if (!inlet.verify()) {
    SLIC_ERROR("Input file failed to verify.");
  }
}

template <>
void defineTestSchema<SolidLegacy>(axom::inlet::Inlet& inlet)
{
  // Integration test parameters
  inlet.addDouble("expected_displacement_l2norm", "Correct L2 norm of the displacement field");
  inlet.addDouble("expected_velocity_l2norm", "Correct L2 norm of the velocity field");

  // Physics
  auto& solid_solver_table = inlet.addStruct("solid", "Finite deformation solid mechanics module");
  // This is the "standard" schema for the actual physics module
  serac::SolidLegacy::InputOptions::defineInputFileSchema(solid_solver_table);

  defineCommonTestSchema(inlet);
}

template <>
void defineTestSchema<ThermalConductionLegacy>(axom::inlet::Inlet& inlet)
{
  // Integration test parameters
  inlet.addDouble("expected_temperature_l2norm", "Correct L2 norm of the temperature field");

  auto& exact = inlet.addStruct("exact_solution", "Exact solution for the temperature field");
  serac::input::CoefficientInputOptions::defineInputFileSchema(exact);

  // Physics
  auto& conduction_table = inlet.addStruct("thermal_conduction", "Thermal conduction module");
  // This is the "standard" schema for the actual physics module
  serac::ThermalConductionLegacy::InputOptions::defineInputFileSchema(conduction_table);

  defineCommonTestSchema(inlet);
}

template <>
void defineTestSchema<ThermalSolidLegacy>(axom::inlet::Inlet& inlet)
{
  // Integration test parameters
  inlet.addDouble("expected_temperature_l2norm", "Correct L2 norm of the temperature field");
  inlet.addDouble("expected_displacement_l2norm", "Correct L2 norm of the displacement field");
  inlet.addDouble("expected_velocity_l2norm", "Correct L2 norm of the velocity field");

  // Physics
  auto& thermal_solid_table = inlet.addStruct("thermal_solid", "Thermal solid module");
  // This is the "standard" schema for the actual physics module
  serac::ThermalSolidLegacy::InputOptions::defineInputFileSchema(thermal_solid_table);

  defineCommonTestSchema(inlet);
}

namespace detail {

// Utility type for use in always-false static assertions
template <typename>
struct AlwaysFalse {
  static constexpr bool value = false;
};

/**
 * @brief Returns the name of the module used by the input file
 */
template <typename PhysicsModule>
std::string moduleName()
{
  static_assert(AlwaysFalse<PhysicsModule>::value, "Test driver is not supported for selected type");
  return {};
}

template <>
std::string moduleName<SolidLegacy>()
{
  return "solid";
}

template <>
std::string moduleName<ThermalConductionLegacy>()
{
  return "thermal_conduction";
}

template <>
std::string moduleName<ThermalSolidLegacy>()
{
  return "thermal_solid";
}

/**
 * @brief Verifies the solution fields against the input file
 * @param[in] phys_module The module whose fields should be verified
 * @param[in] inlet The Inlet object from which expected solution values will be obtained
 */
template <typename PhysicsModule>
void verifyFields(const PhysicsModule&, const axom::inlet::Inlet&)
{
  static_assert(AlwaysFalse<PhysicsModule>::value, "Test driver is not supported for selected type");
}

template <>
void verifyFields(const SolidLegacy& phys_module, const axom::inlet::Inlet& inlet)
{
  if (inlet.contains("expected_displacement_l2norm")) {
    double x_norm = norm(phys_module.displacement());
    EXPECT_NEAR(inlet["expected_displacement_l2norm"], x_norm, inlet["epsilon"]);
  }
  if (inlet.contains("expected_velocity_l2norm")) {
    double v_norm = norm(phys_module.velocity());
    EXPECT_NEAR(inlet["expected_velocity_l2norm"], v_norm, inlet["epsilon"]);
  }
}

template <>
void verifyFields(const ThermalConductionLegacy& phys_module, const axom::inlet::Inlet& inlet)
{
  if (inlet.contains("expected_temperature_l2norm")) {
    EXPECT_NEAR(inlet["expected_temperature_l2norm"], norm(phys_module.temperature()), inlet["epsilon"]);
  }

  if (inlet.contains("exact_solution")) {
    auto coef_options = inlet["exact_solution"].get<serac::input::CoefficientInputOptions>();
    auto exact        = coef_options.constructScalar();
    exact->SetTime(phys_module.time());
    double error = phys_module.temperature().gridFunction().ComputeLpError(2.0, *exact);
    EXPECT_NEAR(error, 0.0, inlet["epsilon"]);
  }
}

template <>
void verifyFields(const ThermalSolidLegacy& phys_module, const axom::inlet::Inlet& inlet)
{
  if (inlet.contains("expected_displacement_l2norm")) {
    double x_norm = norm(phys_module.displacement());
    EXPECT_NEAR(inlet["expected_displacement_l2norm"], x_norm, inlet["epsilon"]);
  }
  if (inlet.contains("expected_velocity_l2norm")) {
    double v_norm = norm(phys_module.velocity());
    EXPECT_NEAR(inlet["expected_velocity_l2norm"], v_norm, inlet["epsilon"]);
  }
  if (inlet.contains("expected_temperature_l2norm")) {
    double t_norm = norm(phys_module.temperature());
    EXPECT_NEAR(inlet["expected_temperature_l2norm"], t_norm, inlet["epsilon"]);
  }
}

}  // namespace detail

template <typename PhysicsModule>
void runModuleTest(const std::string& input_file, const std::string& test_name, std::optional<int> restart_cycle)
{
  // Create DataStore
  axom::sidre::DataStore datastore;

  // Initialize the DataCollection
  // WARNING: This must happen before serac::input::initialize, as the loading
  // process will wipe out the datastore
  std::string output_directory = test_name;
  axom::utilities::filesystem::makeDirsForPath(output_directory);
  serac::StateManager::initialize(datastore, output_directory);
  if (restart_cycle) {
    serac::StateManager::load(*restart_cycle);
  }

  // Initialize Inlet and read input file
  auto inlet = serac::input::initialize(datastore, input_file);

  defineTestSchema<PhysicsModule>(inlet);

  // Build the mesh
  if (!restart_cycle) {
    auto mesh_options = inlet["main_mesh"].get<serac::mesh::InputOptions>();
    if (const auto file_options = std::get_if<serac::mesh::FileInputOptions>(&mesh_options.extra_options)) {
      file_options->absolute_mesh_file_name =
          serac::input::findMeshFilePath(file_options->relative_mesh_file_name, input_file);
    }
    auto mesh = serac::mesh::buildParallelMesh(mesh_options);
    serac::StateManager::setMesh(std::move(mesh));
  }

  const std::string module_name = detail::moduleName<PhysicsModule>();

  // Define the solid solver object
  auto          module_options = inlet[module_name].get<typename PhysicsModule::InputOptions>();
  PhysicsModule phys_module(module_options);

  const bool is_dynamic = inlet[module_name].contains("dynamics");

  // Complete the solver setup
  phys_module.completeSetup();
  // Output the initial state
  phys_module.outputState();

  double dt = inlet["dt"];

  // Check if dynamic
  if (is_dynamic) {
    double t       = 0.0;
    double t_final = inlet["t_final"];

    // Perform time-integration
    // (looping over the time iterations, ti, with a time-step dt).
    bool last_step = false;
    for (int ti = 1; !last_step; ti++) {
      double dt_real = std::min(dt, t_final - t);
      t += dt_real;
      last_step = (t >= t_final - 1e-8 * dt);

      phys_module.advanceTimestep(dt_real);
    }
    phys_module.setTime(t);
  } else {
    phys_module.advanceTimestep(dt);
  }

  // Output the final state
  phys_module.outputState();

  detail::verifyFields(phys_module, inlet);
  // WARNING: This will destroy the mesh before the SolidLegacy module destructor gets called
  // serac::StateManager::reset();
}

template void runModuleTest<SolidLegacy>(const std::string&, const std::string&, std::optional<int>);
template void runModuleTest<ThermalConductionLegacy>(const std::string&, const std::string&, std::optional<int>);
template void runModuleTest<ThermalSolidLegacy>(const std::string&, const std::string&, std::optional<int>);

}  // end namespace test_utils

}  // end namespace serac

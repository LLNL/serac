// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_utilities.hpp"

#include <gtest/gtest.h>

#include "serac/infrastructure/input.hpp"
#include "serac/numerics/mesh_utils.hpp"
#include "serac/physics/nonlinear_solid.hpp"
#include "serac/physics/thermal_conduction.hpp"

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

  inlet.addString("output_type", "Desired output format")
      .validValues({"GLVis", "ParaView", "VisIt", "SidreVisIt"})
      .defaultValue("VisIt");

  // Comparison parameter
  inlet.addDouble("epsilon", "Threshold to be used in the comparison");

  auto& mesh_table = inlet.addTable("main_mesh", "The main mesh for the problem");
  serac::mesh::InputOptions::defineInputFileSchema(mesh_table);

  // Verify input file
  if (!inlet.verify()) {
    SLIC_ERROR("Input file failed to verify.");
  }
}

template <>
void defineTestSchema<NonlinearSolid>(axom::inlet::Inlet& inlet)
{
  // Integration test parameters
  inlet.addDouble("expected_x_l2norm", "Correct L2 norm of the displacement field");
  inlet.addDouble("expected_v_l2norm", "Correct L2 norm of the velocity field");

  // Physics
  auto& solid_solver_table = inlet.addTable("nonlinear_solid", "Finite deformation solid mechanics module");
  // This is the "standard" schema for the actual physics module
  serac::NonlinearSolid::InputOptions::defineInputFileSchema(solid_solver_table);

  defineCommonTestSchema(inlet);
}

template <>
void defineTestSchema<ThermalConduction>(axom::inlet::Inlet& inlet)
{
  // Integration test parameters
  inlet.addDouble("expected_t_l2norm", "Correct L2 norm of the temperature field");

  // Physics
  auto& conduction_table = inlet.addTable("thermal_conduction", "Thermal conduction module");
  // This is the "standard" schema for the actual physics module
  serac::ThermalConduction::InputOptions::defineInputFileSchema(conduction_table);

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
std::string moduleName<NonlinearSolid>()
{
  return "nonlinear_solid";
}

template <>
std::string moduleName<ThermalConduction>()
{
  return "thermal_conduction";
}

/**
 * @brief Verifies the solution fields against the input file
 * @param[in] module The module whose fields should be verified
 * @param[in] inlet The Inlet object from which expected solution values will be obtained
 * @param[in] dim The mesh dimension
 */
template <typename PhysicsModule>
void verifyFields(const PhysicsModule&, const axom::inlet::Inlet&, const int)
{
  static_assert(AlwaysFalse<PhysicsModule>::value, "Test driver is not supported for selected type");
}

template <>
void verifyFields(const NonlinearSolid& module, const axom::inlet::Inlet& inlet, const int dim)
{
  mfem::Vector zero(dim);
  zero = 0.0;
  mfem::VectorConstantCoefficient zerovec(zero);

  if (inlet.contains("expected_x_l2norm")) {
    double x_norm = module.displacement().gridFunc().ComputeLpError(2.0, zerovec);
    EXPECT_NEAR(inlet["expected_x_l2norm"], x_norm, inlet["epsilon"]);
  }
  if (inlet.contains("expected_v_l2norm")) {
    double v_norm = module.velocity().gridFunc().ComputeLpError(2.0, zerovec);
    EXPECT_NEAR(inlet["expected_v_l2norm"], v_norm, inlet["epsilon"]);
  }
}

template <>
void verifyFields(const ThermalConduction& module, const axom::inlet::Inlet& inlet, const int)
{
  mfem::ConstantCoefficient zero(0.0);
  if (inlet.contains("expected_t_l2norm")) {
    double t_norm = module.temperature().gridFunc().ComputeLpError(2.0, zero);
    EXPECT_NEAR(inlet["expected_t_l2norm"], t_norm, inlet["epsilon"]);
  }
}
}  // namespace detail

template <typename PhysicsModule>
void runModuleTest(const std::string& input_file, std::unique_ptr<mfem::ParMesh> custom_mesh)
{
  // Create DataStore
  axom::sidre::DataStore datastore;

  // Initialize Inlet and read input file
  auto inlet = serac::input::initialize(datastore, input_file);
  serac::StateManager::initialize(datastore);

  defineTestSchema<PhysicsModule>(inlet);

  // Build the mesh
  std::unique_ptr<mfem::ParMesh> mesh;
  if (custom_mesh) {
    mesh = std::move(custom_mesh);
  } else {
    auto mesh_options   = inlet["main_mesh"].get<serac::mesh::InputOptions>();
    auto full_mesh_path = serac::input::findMeshFilePath(mesh_options.relative_mesh_file_name, input_file);
    mesh = serac::buildMeshFromFile(full_mesh_path, mesh_options.ser_ref_levels, mesh_options.par_ref_levels);
  }
  const int dim = mesh->Dimension();
  serac::StateManager::setMesh(std::move(mesh));

  const std::string module_name = detail::moduleName<PhysicsModule>();

  // Define the solid solver object
  auto          module_options = inlet[module_name].get<typename PhysicsModule::InputOptions>();
  PhysicsModule module(module_options);

  const bool is_dynamic = inlet[module_name].contains("dynamics");

  // Initialize the output
  const static auto output_names = []() {
    std::unordered_map<std::string, serac::OutputType> result;
    result["GLVis"]      = serac::OutputType::GLVis;
    result["ParaView"]   = serac::OutputType::ParaView;
    result["VisIt"]      = serac::OutputType::VisIt;
    result["SidreVisIt"] = serac::OutputType::SidreVisIt;
    return result;
  }();

  module.initializeOutput(output_names.at(inlet["output_type"]), module_name);

  // Complete the solver setup
  module.completeSetup();
  // Output the initial state
  module.outputState();

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

      module.advanceTimestep(dt_real);
    }
  } else {
    module.advanceTimestep(dt);
  }

  // Output the final state
  module.outputState();

  detail::verifyFields(module, inlet, dim);
  serac::StateManager::reset();
}

template void runModuleTest<NonlinearSolid>(const std::string& input_file, std::unique_ptr<mfem::ParMesh> custom_mesh);
template void runModuleTest<ThermalConduction>(const std::string&             input_file,
                                               std::unique_ptr<mfem::ParMesh> custom_mesh);

}  // end namespace test_utils

}  // end namespace serac

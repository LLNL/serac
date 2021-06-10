// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file serac.cpp
 *
 * @brief Serac: nonlinear implicit thermal-structural driver
 *
 * The purpose of this code is to act as a proxy app for nonlinear implicit mechanics codes at LLNL.
 */

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "axom/core.hpp"
#include "mfem.hpp"
#include "serac/coefficients/loading_functions.hpp"
#include "serac/infrastructure/cli.hpp"
#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/input.hpp"
#include "serac/infrastructure/logger.hpp"
#include "serac/infrastructure/output.hpp"
#include "serac/infrastructure/terminator.hpp"
#include "serac/numerics/mesh_utils.hpp"
#include "serac/physics/thermal_solid.hpp"
#include "serac/physics/utilities/equation_solver.hpp"
#include "serac/serac_config.hpp"

namespace serac {

/**
 * @brief Define the input file structure for the driver code
 *
 * @param[in] inlet The inlet instance
 */
void defineInputFileSchema(axom::inlet::Inlet& inlet)
{
  // Simulation time parameters
  inlet.addDouble("t_final", "Final time for simulation.").defaultValue(1.0);
  inlet.addDouble("dt", "Time step.").defaultValue(0.25);

  // The output type (visit, glvis, paraview, etc)
  serac::input::defineOutputTypeInputFileSchema(inlet.getGlobalContainer());

  // The mesh options
  auto& mesh_table = inlet.addStruct("main_mesh", "The main mesh for the problem");
  serac::mesh::InputOptions::defineInputFileSchema(mesh_table);

  // The solid mechanics options
  auto& solid_solver_table = inlet.addStruct("solid", "Finite deformation solid mechanics module");
  serac::Solid::InputOptions::defineInputFileSchema(solid_solver_table);

  // The thermal conduction options
  auto& thermal_solver_table = inlet.addStruct("thermal_conduction", "Thermal conduction module");
  serac::ThermalConduction::InputOptions::defineInputFileSchema(thermal_solver_table);

  // Verify the input file
  if (!inlet.verify()) {
    SLIC_ERROR_ROOT("Input file failed to verify.");
  }
}

}  // namespace serac

/**
 * @brief The main serac driver code
 *
 * @param[in] argc Number of input arguments
 * @param[in] argv The vector of input arguments
 *
 * @return The return code
 */
int main(int argc, char* argv[])
{
  serac::initialize(argc, argv);

  // Handle Command line
  std::unordered_map<std::string, std::string> cli_opts =
      serac::cli::defineAndParse(argc, argv, "Serac: a high order nonlinear thermomechanical simulation code");
  serac::cli::printGiven(cli_opts);

  // Read input file
  std::string input_file_path = "";
  auto        search          = cli_opts.find("input_file");
  if (search != cli_opts.end()) {
    input_file_path = search->second;
  }

  // Check if a restart was requested
  std::optional<int> restart_cycle;
  if (auto cycle = cli_opts.find("restart_cycle"); cycle != cli_opts.end()) {
    restart_cycle = std::stoi(cycle->second);
  }

  // Check for the doc creation command line argument
  bool create_input_file_docs = cli_opts.find("create_input_file_docs") != cli_opts.end();
  // Check for the output fields command line argument
  bool output_fields = cli_opts.find("output_fields") != cli_opts.end();

  // Create DataStore
  axom::sidre::DataStore datastore;

  // Intialize MFEMSidreDataCollection
  // If restart_cycle is non-empty, then this is a restart run and the data will be loaded here
  serac::StateManager::initialize(datastore, restart_cycle);

  // Initialize Inlet and read input file
  auto inlet = serac::input::initialize(datastore, input_file_path);
  serac::defineInputFileSchema(inlet);

  // Optionally, create input file documentation and quit
  if (create_input_file_docs) {
    auto writer = std::make_unique<axom::inlet::SphinxWriter>("serac_input.rst");
    inlet.registerWriter(std::move(writer));
    inlet.write();
    serac::exitGracefully();
  }

  // Save input values to file
  datastore.getRoot()->getGroup("input_file")->save("serac_input.json", "json");

  // Not restarting, so we need to create the mesh and register it with the StateManager
  if (!restart_cycle) {
    // Build the mesh
    auto mesh_options = inlet["main_mesh"].get<serac::mesh::InputOptions>();
    if (const auto file_opts = std::get_if<serac::mesh::FileInputOptions>(&mesh_options.extra_options)) {
      file_opts->absolute_mesh_file_name =
          serac::input::findMeshFilePath(file_opts->relative_mesh_file_name, input_file_path);
    }
    auto mesh = serac::mesh::buildParallelMesh(mesh_options);
    serac::StateManager::setMesh(std::move(mesh));
  }

  // Create the physics object
  std::unique_ptr<serac::BasePhysics> main_physics;

  // Create nullable contains for the solid and thermal input file options
  std::optional<serac::Solid::InputOptions>             solid_solver_options;
  std::optional<serac::ThermalConduction::InputOptions> thermal_solver_options;

  // If the blocks exist, read the appropriate input file options
  // FIXME: This will get patched in inlet so we can do inlet.isUserProvided("solid") etc
  if (inlet.getGlobalContainer().getChildContainers().at("solid")->isUserProvided()) {
    solid_solver_options = inlet["solid"].get<serac::Solid::InputOptions>();
  }
  if (inlet.getGlobalContainer().getChildContainers().at("thermal_conduction")->isUserProvided()) {
    thermal_solver_options = inlet["thermal_conduction"].get<serac::ThermalConduction::InputOptions>();
  }

  // Construct the appropriate physics object using the input file options
  if (solid_solver_options && thermal_solver_options) {
    main_physics = std::make_unique<serac::ThermalSolid>(*thermal_solver_options, *solid_solver_options);
  } else if (solid_solver_options) {
    main_physics = std::make_unique<serac::Solid>(*solid_solver_options);
  } else if (thermal_solver_options) {
    main_physics = std::make_unique<serac::ThermalConduction>(*thermal_solver_options);
  } else {
    SLIC_ERROR_ROOT("Neither solid nor thermal_conduction blocks specified in the input file.");
  }

  // Complete the solver setup
  main_physics->completeSetup();

  // Initialize/set the time information
  double t       = 0;
  double t_final = inlet["t_final"];
  double dt      = inlet["dt"];

  bool last_step = false;

  // FIXME: This and the FromInlet specialization are hacked together,
  // should be inlet["output_type"].get<OutputType>()
  main_physics->initializeOutput(inlet.getGlobalContainer().get<serac::OutputType>(), "serac");

  // Enter the time step loop.
  for (int ti = 1; !last_step; ti++) {
    // Compute the real timestep. This may be less than dt for the last timestep.
    double dt_real = std::min(dt, t_final - t);

    // Compute current time
    t = t + dt_real;

    // Print the timestep information
    SLIC_INFO_ROOT("step " << ti << ", t = " << t);

    // Solve the physics module appropriately
    main_physics->advanceTimestep(dt_real);

    // Output a visualization file
    main_physics->outputState();

    // Determine if this is the last timestep
    last_step = (t >= t_final - 1e-8 * dt);
  }

  if (output_fields) {
    serac::output::outputFields(datastore, "serac_fields", t, serac::output::Language::JSON);
  }

  serac::exitGracefully();
}

// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
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

#include "serac/infrastructure/about.hpp"
#include "serac/infrastructure/cli.hpp"
#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/input.hpp"
#include "serac/infrastructure/logger.hpp"
#include "serac/infrastructure/output.hpp"
#include "serac/infrastructure/terminator.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/thermomechanics.hpp"
#include "serac/numerics/equation_solver.hpp"
#include "serac/physics/state/state_manager.hpp"
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

  // The mesh options
  auto& mesh_table = inlet.addStruct("main_mesh", "The main mesh for the problem");
  serac::mesh::InputOptions::defineInputFileSchema(mesh_table);

  // The solid mechanics options
  auto& solid_solver_table = inlet.addStruct("solid", "Finite deformation solid mechanics module");
  serac::SolidLegacy::InputOptions::defineInputFileSchema(solid_solver_table);

  // The thermal conduction options
  auto& thermal_solver_table = inlet.addStruct("thermal_conduction", "Thermal conduction module");
  serac::ThermalConductionLegacy::InputOptions::defineInputFileSchema(thermal_solver_table);

  // The thermal solid options
  auto& thermal_solid_solver_table = inlet.addStruct("thermal_solid", "Thermal solid module");
  serac::ThermalSolidLegacy::InputOptions::defineInputFileSchema(thermal_solid_solver_table);

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

  // Optionally, print about info and quit
  // TODO: add option for just version and a longer for about?
  bool print_version = cli_opts.find("version") != cli_opts.end();
  if (print_version) {
    SLIC_INFO(serac::about());
    serac::exitGracefully();
  }

  // Output helpful run information
  serac::printRunInfo();
  serac::cli::printGiven(cli_opts);

  // Read input file
  std::string input_file_path = "";
  auto        search          = cli_opts.find("input-file");
  if (search != cli_opts.end()) {
    input_file_path = search->second;
  }

  // Output directory used for all files written to the file system.
  // Example of outputted files:
  //
  // * Inlet docs + input file value file
  // * StateManager state files
  // * Summary file
  std::string output_directory = "";
  search                       = cli_opts.find("output-directory");
  if (search != cli_opts.end()) {
    output_directory = search->second;
  }
  axom::utilities::filesystem::makeDirsForPath(output_directory);

  search = cli_opts.find("paraview-directory");

  std::optional<std::string> paraview_output_dir = {};
  if (search != cli_opts.end()) {
    paraview_output_dir = search->second;
    axom::utilities::filesystem::makeDirsForPath(*paraview_output_dir);
  }

  // Check if a restart was requested
  std::optional<int> restart_cycle;
  if (auto cycle = cli_opts.find("restart-cycle"); cycle != cli_opts.end()) {
    restart_cycle = std::stoi(cycle->second);
  }

  // Create DataStore
  axom::sidre::DataStore datastore;

  // Intialize MFEMSidreDataCollection
  serac::StateManager::initialize(datastore, output_directory);

  // Initialize Inlet and read input file
  auto inlet = serac::input::initialize(datastore, input_file_path);
  serac::defineInputFileSchema(inlet);

  // Optionally, create input file documentation and quit
  bool create_input_file_docs = cli_opts.find("create-input-file-docs") != cli_opts.end();
  if (create_input_file_docs) {
    std::string input_docs_path = axom::utilities::filesystem::joinPath(output_directory, "serac_input.rst");
    inlet.write(axom::inlet::SphinxWriter(input_docs_path));
    serac::exitGracefully();
  }

  // Save input values to file
  std::string input_values_path = axom::utilities::filesystem::joinPath(output_directory, "serac_input_values.json");
  datastore.getRoot()->getGroup("input_file")->save(input_values_path, "json");

  // Initialize/set the time information
  double t       = 0;
  double t_final = inlet["t_final"];
  double dt      = inlet["dt"];
  int    cycle   = 1;

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
  } else {
    // If restart_cycle is non-empty, then this is a restart run and the data will be loaded here
    t     = serac::StateManager::load(*restart_cycle);
    cycle = *restart_cycle;
  }

  // Create the physics object
  std::unique_ptr<serac::BasePhysics> main_physics;

  // Create nullable containers for the solid and thermal input file options
  std::optional<serac::SolidMechanics::InputOptions>          solid_solver_options;
  // TODO: Fix these 
  std::optional<serac::ThermalConductionLegacy::InputOptions> thermal_solver_options;
  std::optional<serac::ThermalSolidLegacy::InputOptions>      thermal_solid_solver_options;

  // If the blocks exist, read the appropriate input file options
  if (inlet.isUserProvided("solid")) {
    solid_solver_options = inlet["solid"].get<serac::SolidLegacy::InputOptions>();
  }
  // TODO: Fix these 
  // if (inlet.isUserProvided("thermal_conduction")) {
  //   thermal_solver_options = inlet["thermal_conduction"].get<serac::ThermalConductionLegacy::InputOptions>();
  // }
  // if (inlet.isUserProvided("thermal_solid")) {
  //   thermal_solid_solver_options = inlet["thermal_solid"].get<serac::ThermalSolidLegacy::InputOptions>();
  // }

  // Construct the appropriate physics object using the input file options
  auto dim = serac::StateManager::mesh().Dimension();
  SLIC_ERROR_IF_ROOT(dim < 2 || dim > 3, axom::fmt::format("Invalid mesh dimension '{0}' provided. Valid values are 2 or 3.", dim);
  if (thermal_solid_solver_options) {
    main_physics = std::make_unique<serac::ThermalSolidLegacy>(*thermal_solid_solver_options);
  } else if (solid_solver_options && thermal_solver_options) {
    main_physics = std::make_unique<serac::ThermalSolidLegacy>(*thermal_solver_options, *solid_solver_options);
  } else if (solid_solver_options) {
    int order = solid_solver_options->order;
    SLIC_ERROR_IF_ROOT(order < 1 || order > 3, "Invalid Solid solver order '{0}' given. Valid values are 1, 2, or 3.", order);
    if (solid_solver_options->order == 1) {
      if (dim == 2) {
        main_physics = std::make_unique<serac::SolidMechanics<1, 2>>(*solid_solver_options);
      } else if (dim == 3) {
        main_physics = std::make_unique<serac::SolidMechanics<1, 3>>(*solid_solver_options);
      }
    } else if (solid_solver_options->order == 2) {
      if (dim == 2) {
        main_physics = std::make_unique<serac::SolidMechanics<2, 2>>(*solid_solver_options);
      } else if (dim == 3) {
        main_physics = std::make_unique<serac::SolidMechanics<2, 3>>(*solid_solver_options);
      }
    } else if (solid_solver_options->order == 3) {
      if  (dim == 2) {
        main_physics = std::make_unique<serac::SolidMechanics<3, 2>>(*solid_solver_options);
      } else if (dim == 3) {
        main_physics = std::make_unique<serac::SolidMechanics<3, 3>>(*solid_solver_options);
      }
  } else if (thermal_solver_options) {
    main_physics = std::make_unique<serac::ThermalConductionLegacy>(*thermal_solver_options);
  } else {
    SLIC_ERROR_ROOT("Neither solid, thermal_conduction, nor thermal_solid blocks specified in the input file.");
  }

  // Complete the solver setup
  main_physics->completeSetup();

  // Update physics time and cycle
  main_physics->setTime(t);
  main_physics->setCycle(cycle);

  main_physics->initializeSummary(datastore, t_final, dt);

  // Enter the time step loop.
  bool last_step = false;
  while (!last_step) {
    // Flush all messages held by the logger
    serac::logger::flush();

    // Compute the real timestep. This may be less than dt for the last timestep.
    double dt_real = std::min(dt, t_final - t);

    // Compute current time
    t = t + dt_real;

    // Print the timestep information
    SLIC_INFO_ROOT("step " << cycle << ", t = " << t);

    // Solve the physics module appropriately
    main_physics->advanceTimestep(dt_real);

    // Output a visualization file
    main_physics->outputState(paraview_output_dir);

    // Save curve data to Sidre datastore to be output later
    main_physics->saveSummary(datastore, t);

    // Determine if this is the last timestep
    last_step = (t >= t_final - 1e-8 * dt);

    // Increment cycle
    cycle++;
  }

  // Output summary file (basic run info and curve data)
  serac::output::outputSummary(datastore, output_directory);

  serac::exitGracefully();
}

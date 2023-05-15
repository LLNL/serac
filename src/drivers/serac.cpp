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
#include "serac/physics/solid_mechanics.hpp"
#include "serac/physics/heat_transfer.hpp"
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
  SolidMechanicsInputOptions::defineInputFileSchema(solid_solver_table);

  // The thermal conduction options
  // TODO: rename to 'heat_transfer'?
  auto& thermal_solver_table = inlet.addStruct("thermal_conduction", "Thermal conduction module");
  HeatTransferInputOptions::defineInputFileSchema(thermal_solver_table);

  // The thermal solid options
  auto& thermal_solid_solver_table = inlet.addStruct("thermal_solid", "Thermal solid module");
  ThermomechanicsInputOptions::defineInputFileSchema(thermal_solid_solver_table);

  // Verify the input file
  if (!inlet.verify()) {
    SLIC_ERROR_ROOT("Input file failed to verify.");
  }
}

}  // namespace serac

/**
 * @brief Constructs the appropriate physics object using the input file options
 *
 * @param[in] order The order of the discretization
 * @param[in] dim The spatial dimension of the mesh
 * @param[in] solid_mechanics_options Optional container of input options for SolidMechanics physics module
 * @param[in] heat_transfer_options   Optional container of input options for HeatTransfer physics module
 * @param[in] thermomechanics_options Optional container of input options for Thermomechanics physics module
 *
 * @return Base class instance of the created physics class
 */
std::unique_ptr<serac::BasePhysics> createPhysics(
    int dim, int order, std::optional<serac::SolidMechanicsInputOptions> solid_mechanics_options,
    std::optional<serac::HeatTransferInputOptions>    heat_transfer_options,
    std::optional<serac::ThermomechanicsInputOptions> thermomechanics_options)
{
  std::unique_ptr<serac::BasePhysics> main_physics;
  if (thermomechanics_options) {
    if (order == 1) {
      if (dim == 2) {
        main_physics = std::make_unique<serac::Thermomechanics<1, 2>>(*thermomechanics_options);
      } else if (dim == 3) {
        main_physics = std::make_unique<serac::Thermomechanics<1, 3>>(*thermomechanics_options);
      }
    } else if (order == 2) {
      if (dim == 2) {
        main_physics = std::make_unique<serac::Thermomechanics<2, 2>>(*thermomechanics_options);
      } else if (dim == 3) {
        main_physics = std::make_unique<serac::Thermomechanics<2, 3>>(*thermomechanics_options);
      }
    } else if (order == 3) {
      if (dim == 2) {
        main_physics = std::make_unique<serac::Thermomechanics<3, 2>>(*thermomechanics_options);
      } else if (dim == 3) {
        main_physics = std::make_unique<serac::Thermomechanics<3, 3>>(*thermomechanics_options);
      }
    }
  } else if (solid_mechanics_options && heat_transfer_options) {
    if (order == 1) {
      if (dim == 2) {
        main_physics = std::make_unique<serac::Thermomechanics<1, 2>>(*heat_transfer_options, *solid_mechanics_options);
      } else if (dim == 3) {
        main_physics = std::make_unique<serac::Thermomechanics<1, 3>>(*heat_transfer_options, *solid_mechanics_options);
      }
    } else if (order == 2) {
      if (dim == 2) {
        main_physics = std::make_unique<serac::Thermomechanics<2, 2>>(*heat_transfer_options, *solid_mechanics_options);
      } else if (dim == 3) {
        main_physics = std::make_unique<serac::Thermomechanics<2, 3>>(*heat_transfer_options, *solid_mechanics_options);
      }
    } else if (order == 3) {
      if (dim == 2) {
        main_physics = std::make_unique<serac::Thermomechanics<3, 2>>(*heat_transfer_options, *solid_mechanics_options);
      } else if (dim == 3) {
        main_physics = std::make_unique<serac::Thermomechanics<3, 3>>(*heat_transfer_options, *solid_mechanics_options);
      }
    }
  } else if (solid_mechanics_options) {
    if (order == 1) {
      if (dim == 2) {
        main_physics = std::make_unique<serac::SolidMechanics<1, 2>>(*solid_mechanics_options);
      } else if (dim == 3) {
        main_physics = std::make_unique<serac::SolidMechanics<1, 3>>(*solid_mechanics_options);
      }
    } else if (order == 2) {
      if (dim == 2) {
        main_physics = std::make_unique<serac::SolidMechanics<2, 2>>(*solid_mechanics_options);
      } else if (dim == 3) {
        main_physics = std::make_unique<serac::SolidMechanics<2, 3>>(*solid_mechanics_options);
      }
    } else if (order == 3) {
      if (dim == 2) {
        main_physics = std::make_unique<serac::SolidMechanics<3, 2>>(*solid_mechanics_options);
      } else if (dim == 3) {
        main_physics = std::make_unique<serac::SolidMechanics<3, 3>>(*solid_mechanics_options);
      }
    }
  } else if (heat_transfer_options) {
    if (order == 1) {
      if (dim == 2) {
        main_physics = std::make_unique<serac::HeatTransfer<1, 2>>(*heat_transfer_options);
      } else if (dim == 3) {
        main_physics = std::make_unique<serac::HeatTransfer<1, 3>>(*heat_transfer_options);
      }
    } else if (order == 2) {
      if (dim == 2) {
        main_physics = std::make_unique<serac::HeatTransfer<2, 2>>(*heat_transfer_options);
      } else if (dim == 3) {
        main_physics = std::make_unique<serac::HeatTransfer<2, 3>>(*heat_transfer_options);
      }
    } else if (order == 3) {
      if (dim == 2) {
        main_physics = std::make_unique<serac::HeatTransfer<3, 2>>(*heat_transfer_options);
      } else if (dim == 3) {
        main_physics = std::make_unique<serac::HeatTransfer<3, 3>>(*heat_transfer_options);
      }
    }
  } else {
    SLIC_ERROR_ROOT("Neither solid, thermal_conduction, nor thermal_solid blocks specified in the input file.");
  }
  return main_physics;
}

/**
 * @brief Return and check correctness of the order of discretization
 *
 * @param[in] solid_mechanics_options Optional container of input options for SolidMechanics physics module
 * @param[in] heat_transfer_options   Optional container of input options for HeatTransfer physics module
 * @param[in] thermomechanics_options Optional container of input options for Thermomechanics physics module
 *
 * @return The order of the discretization
 */
int getOrder(std::optional<serac::SolidMechanicsInputOptions>  solid_mechanics_options,
             std::optional<serac::HeatTransferInputOptions>    heat_transfer_options,
             std::optional<serac::ThermomechanicsInputOptions> thermomechanics_options)
{
  int order = 0;
  if (thermomechanics_options) {
    order             = thermomechanics_options->solid_options.order;
    int thermal_order = thermomechanics_options->thermal_options.order;
    SLIC_ERROR_ROOT_IF(
        order != thermal_order,
        axom::fmt::format("Solid order '{0}' and thermal order '{1}'' do not match.", order, thermal_order));
  } else if (solid_mechanics_options && heat_transfer_options) {
    order             = solid_mechanics_options->order;
    int thermal_order = heat_transfer_options->order;
    SLIC_ERROR_ROOT_IF(
        order != thermal_order,
        axom::fmt::format("Solid order '{0}' and thermal order '{1}'' do not match.", order, thermal_order));
  } else if (solid_mechanics_options) {
    order = solid_mechanics_options->order;
  } else if (heat_transfer_options) {
    order = heat_transfer_options->order;
  } else {
    SLIC_ERROR_ROOT("Neither solid, thermal_conduction, nor thermal_solid blocks specified in the input file.");
  }
  SLIC_ERROR_ROOT_IF(order < 1 || order > 3,
                     axom::fmt::format("Invalid solver order '{0}' given. Valid values are 1, 2, or 3.", order));
  return order;
}

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

  // TODO: Remove this after input file functionality is restored.
  const std::string warning_message = R"PREFIX(******************************
This Serac driver does not
currently work due to missing
input file functionality.
******************************)PREFIX";
  SLIC_INFO(warning_message);

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

  // Create nullable containers for the solid and thermal input file options
  std::optional<serac::SolidMechanicsInputOptions>  solid_mechanics_options;
  std::optional<serac::HeatTransferInputOptions>    heat_transfer_options;
  std::optional<serac::ThermomechanicsInputOptions> thermomechanics_options;

  // If the blocks exist, read the appropriate input file options
  if (inlet.isUserProvided("solid")) {
    solid_mechanics_options = inlet["solid"].get<serac::SolidMechanicsInputOptions>();
  }
  if (inlet.isUserProvided("thermal_conduction")) {
    heat_transfer_options = inlet["thermal_conduction"].get<serac::HeatTransferInputOptions>();
  }
  if (inlet.isUserProvided("thermal_solid")) {
    thermomechanics_options = inlet["thermal_solid"].get<serac::ThermomechanicsInputOptions>();
  }

  // Get dimension and order of problem
  int dim = serac::StateManager::mesh().Dimension();
  SLIC_ERROR_ROOT_IF(dim < 2 || dim > 3,
                     axom::fmt::format("Invalid mesh dimension '{0}' provided. Valid values are 2 or 3.", dim));
  int order = getOrder(solid_mechanics_options, heat_transfer_options, thermomechanics_options);

  // Create the physics object
  auto main_physics =
      createPhysics(dim, order, solid_mechanics_options, heat_transfer_options, thermomechanics_options);

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

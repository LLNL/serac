// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file with_input_file.cpp
 *
 * @brief A simple example of steady-state thermal conduction that uses
 * a Lua input file to configure the simulation
 */

#include "serac/physics/thermal_conduction.hpp"
#include "serac/physics/utilities/state_manager.hpp"
#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/terminator.hpp"
#include "serac/numerics/mesh_utils.hpp"
#include "serac/serac_config.hpp" // for SERAC_REPO_DIR

const auto input_file = SERAC_REPO_DIR "/examples/simple_conduction/conduction.lua";

int main(int argc, char* argv[])
{
  /*auto [num_procs, rank] = */serac::initialize(argc, argv);

  // Initialize the data store and input file reader
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore);
  auto inlet = serac::input::initialize(datastore, input_file);

  // Read the mesh options from the input file via inlet
  auto& mesh_schema = inlet.addStruct("main_mesh", "The main mesh for the problem");
  serac::mesh::InputOptions::defineInputFileSchema(mesh_schema);

  // Read the thermal conduction options from the input file via inlet
  auto& thermal_schema = inlet.addStruct("thermal_conduction", "Thermal conduction module");
  serac::ThermalConduction::InputOptions::defineInputFileSchema(thermal_schema);

  // Read the output type from the input file via inlet
  serac::input::defineOutputTypeInputFileSchema(inlet.getGlobalContainer());

  // Verify that the input file parsed correctly
  SLIC_ERROR_ROOT_IF(!inlet.verify(), "Input file contained errors");

  // Build the mesh and pass it to the state manager
  auto mesh_options = inlet["main_mesh"].get<serac::mesh::InputOptions>();
  auto mesh = serac::mesh::buildParallelMesh(mesh_options);
  serac::StateManager::setMesh(std::move(mesh));

  // Build a thermal conduction physics module from the input file options
  auto conduction_opts = inlet["thermal_conduction"].get<serac::ThermalConduction::InputOptions>();
  serac::ThermalConduction conduction(conduction_opts);

  // Initialize the output files
  conduction.initializeOutput(inlet.getGlobalContainer().get<serac::OutputType>(), "simple_conduction_with_input_file");

  // Finalize the MFEM-based data structures inside the thermal conduction module
  conduction.completeSetup();

  // Output the initial state to the chosen file type
  conduction.outputState();

  // This solves the PDE system. As the given input file is quasi-static, this call
  // performs a single solve.
  double dt = 1.0;
  conduction.advanceTimestep(dt);

  // Output the final state
  conduction.outputState();

  // Clean up all of the software infrastructure
  serac::exitGracefully();
}

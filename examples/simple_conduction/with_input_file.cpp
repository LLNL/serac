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
#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/terminator.hpp"
#include "serac/numerics/mesh_utils.hpp"
#include "serac/serac_config.hpp" // for SERAC_REPO_DIR

const auto input_file = SERAC_REPO_DIR "/examples/simple_conduction/conduction.lua";

int main(int argc, char* argv[])
{
  /*auto [num_procs, rank] = */serac::initialize(argc, argv);

  // _inlet_init_start
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore);
  auto inlet = serac::input::initialize(datastore, input_file);
  // _inlet_init_end

  // _inlet_schema_start
  auto& mesh_schema = inlet.addStruct("main_mesh", "The main mesh for the problem");
  serac::mesh::InputOptions::defineInputFileSchema(mesh_schema);

  auto& thermal_schema = inlet.addStruct("thermal_conduction", "Thermal conduction module");
  serac::ThermalConduction::InputOptions::defineInputFileSchema(thermal_schema);

  serac::input::defineOutputTypeInputFileSchema(inlet.getGlobalContainer());
  // _inlet_schema_end

  // _inlet_verify_start
  SLIC_ERROR_ROOT_IF(!inlet.verify(), "Input file contained errors");
  // _inlet_verify_end

  // _create_mesh_start
  auto mesh_options = inlet["main_mesh"].get<serac::mesh::InputOptions>();
  auto mesh = serac::mesh::buildParallelMesh(mesh_options);
  serac::StateManager::setMesh(std::move(mesh));
  // _create_mesh_end

  // _create_module_start
  auto conduction_opts = inlet["thermal_conduction"].get<serac::ThermalConduction::InputOptions>();
  serac::ThermalConduction conduction(conduction_opts);
  // _create_module_end
  // _output_type_start
  conduction.initializeOutput(inlet.getGlobalContainer().get<serac::OutputType>(), "simple_conduction_with_input_file");
  // _output_type_end
  // Complete the solver setup
  conduction.completeSetup();
  // Output the initial state
  conduction.outputState();

  double dt; // Unused for steady-state simulations
  conduction.advanceTimestep(dt);

  // Output the final state
  conduction.outputState();

  serac::exitGracefully();
}

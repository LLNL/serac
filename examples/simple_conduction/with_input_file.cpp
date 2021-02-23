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

#include "serac/serac_config.hpp" // for SERAC_REPO_DIR
#include "serac/physics/thermal_conduction.hpp" // for serac's thermal conduction module
#include "serac/infrastructure/initialize.hpp" // for serac::initialize
#include "serac/infrastructure/terminator.hpp" // for serac::exitGracefully
#include "serac/numerics/mesh_utils.hpp" // for serac::buildRectangleMesh

const auto input_file = SERAC_REPO_DIR "/examples/simple_conduction/conduction.lua";

int main(int argc, char* argv[])
{
  /*auto [num_procs, rank] = */serac::initialize(argc, argv);

  // Create DataStore
  axom::sidre::DataStore datastore;

  // Initialize Inlet and read input file
  auto inlet = serac::input::initialize(datastore, input_file);

  auto& mesh_schema = inlet.addStruct("main_mesh", "The main mesh for the problem");
  serac::mesh::InputOptions::defineInputFileSchema(mesh_schema);

  auto& thermal_schema = inlet.addStruct("thermal_conduction", "Thermal conduction module");
  serac::ThermalConduction::InputOptions::defineInputFileSchema(thermal_schema);

  auto mesh = serac::buildRectangleMesh(10, 10);

  serac::exitGracefully();
}

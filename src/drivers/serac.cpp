// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file serac.cpp
 *
 * @brief Nonlinear implicit proxy app driver
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
#include "serac/infrastructure/terminator.hpp"
#include "serac/numerics/mesh_utils.hpp"
#include "serac/physics/solid.hpp"
#include "serac/physics/utilities/equation_solver.hpp"
#include "serac/serac_config.hpp"

namespace serac {

//------- Input file -------

void defineInputFileSchema(axom::inlet::Inlet& inlet, int rank)
{
  // Simulation time parameters
  inlet.addDouble("t_final", "Final time for simulation.").defaultValue(1.0);
  inlet.addDouble("dt", "Time step.").defaultValue(0.25);
  serac::input::defineOutputTypeInputFileSchema(inlet.getGlobalTable());

  auto& mesh_table = inlet.addStruct("main_mesh", "The main mesh for the problem");
  serac::mesh::InputOptions::defineInputFileSchema(mesh_table);

  // Physics
  auto& solid_solver_table = inlet.addStruct("solid", "Finite deformation solid mechanics module");
  serac::Solid::InputOptions::defineInputFileSchema(solid_solver_table);

  // Verify input file
  if (!inlet.verify()) {
    SLIC_ERROR_ROOT(rank, "Input file failed to verify.");
  }
}

}  // namespace serac

int main(int argc, char* argv[])
{
  auto [num_procs, rank] = serac::initialize(argc, argv);

  // Handle Command line
  std::unordered_map<std::string, std::string> cli_opts =
      serac::cli::defineAndParse(argc, argv, rank, "Serac: a high order nonlinear thermomechanical simulation code");
  serac::cli::printGiven(cli_opts, rank);

  // Read input file
  std::string input_file_path = "";
  auto        search          = cli_opts.find("input_file");
  if (search != cli_opts.end()) {
    input_file_path = search->second;
  }
  bool create_input_file_docs = cli_opts.find("create_input_file_docs") != cli_opts.end();

  // Create DataStore
  axom::sidre::DataStore datastore;

  // Initialize Inlet and read input file
  auto inlet = serac::input::initialize(datastore, input_file_path);
  serac::defineInputFileSchema(inlet, rank);
  if (create_input_file_docs) {
    auto writer = std::make_unique<axom::inlet::SphinxDocWriter>("serac_input.rst", inlet.sidreGroup());
    inlet.registerDocWriter(std::move(writer));
    inlet.writeDoc();
    serac::exitGracefully();
  }

  // Save input values to file
  datastore.getRoot()->save("serac_input.json", "json");

  std::shared_ptr<mfem::ParMesh> mesh;
  // Build the mesh
  auto mesh_options = inlet["main_mesh"].get<serac::mesh::InputOptions>();
  if (const auto file_opts = std::get_if<serac::mesh::FileInputOptions>(&mesh_options.extra_options)) {
    auto full_mesh_path = serac::input::findMeshFilePath(file_opts->relative_mesh_file_name, input_file_path);
    mesh = serac::buildMeshFromFile(full_mesh_path, mesh_options.ser_ref_levels, mesh_options.par_ref_levels);
  }

  // Define the solid solver object
  auto         solid_solver_options = inlet["solid"].get<serac::Solid::InputOptions>();
  serac::Solid solid_solver(mesh, solid_solver_options);

  // Complete the solver setup
  solid_solver.completeSetup();

  // initialize/set the time
  double t       = 0;
  double t_final = inlet["t_final"];  // has default value
  double dt      = inlet["dt"];       // has default value

  bool last_step = false;

  // FIXME: This and the FromInlet specialization are hacked together,
  // should be inlet["output_type"].get<OutputType>()
  solid_solver.initializeOutput(inlet.getGlobalTable().get<serac::OutputType>(), "serac");

  // enter the time step loop. This was modeled after example 10p.
  for (int ti = 1; !last_step; ti++) {
    double dt_real = std::min(dt, t_final - t);
    // compute current time
    t = t + dt_real;

    SLIC_INFO_ROOT(rank, "step " << ti << ", t = " << t);

    // Solve the Newton system
    solid_solver.advanceTimestep(dt_real);

    solid_solver.outputState();

    last_step = (t >= t_final - 1e-8 * dt);
  }

  serac::exitGracefully();
}

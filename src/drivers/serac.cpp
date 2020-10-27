// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
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
#include "coefficients/loading_functions.hpp"
#include "coefficients/traction_coefficient.hpp"
#include "infrastructure/cli.hpp"
#include "infrastructure/initialize.hpp"
#include "infrastructure/input.hpp"
#include "infrastructure/logger.hpp"
#include "infrastructure/terminator.hpp"
#include "mfem.hpp"
#include "numerics/mesh_utils.hpp"
#include "physics/nonlinear_solid.hpp"
#include "physics/utilities/equation_solver.hpp"
#include "serac_config.hpp"

namespace serac {

//------- Input file -------

void defineInputFileSchema(std::shared_ptr<axom::inlet::Inlet> inlet, int rank)
{
  // Simulation time parameters
  inlet->addDouble("t_final", "Final time for simulation.")->defaultValue(1.0);
  inlet->addDouble("dt", "Time step.")->defaultValue(0.25);

  auto mesh_table = inlet->addTable("main_mesh", "The main mesh for the problem");
  serac::mesh::InputInfo::defineInputFileSchema(*mesh_table);

  // Physics
  auto solid_solver_table = inlet->addTable("nonlinear_solid", "Finite deformation solid mechanics module");
  serac::NonlinearSolid::InputInfo::defineInputFileSchema(*solid_solver_table);

  // Verify input file
  if (!inlet->verify()) {
    SLIC_ERROR_ROOT(rank, "Input file failed to verify.");
    serac::exitGracefully(true);
  }
}

}  // namespace serac

int main(int argc, char* argv[])
{
  auto [num_procs, rank] = serac::initialize(argc, argv);

  // Handle Command line
  std::unordered_map<std::string, std::string> cli_opts = serac::cli::defineAndParse(argc, argv, rank);
  serac::cli::printGiven(cli_opts, rank);

  // Read input file
  std::string input_file_path = "";
  auto        search          = cli_opts.find("input_file");
  if (search != cli_opts.end()) {
    input_file_path = search->second;
  }

  // Create DataStore
  axom::sidre::DataStore datastore;

  // Initialize Inlet and read input file
  auto inlet = serac::input::initialize(datastore, input_file_path);
  serac::defineInputFileSchema(inlet, rank);

  // Save input values to file
  datastore.getRoot()->save("serac_input.json", "json");

  // Build the mesh
  auto mesh_info      = (*inlet)["main_mesh"].get<serac::mesh::InputInfo>();
  auto full_mesh_path = serac::input::findMeshFilePath(mesh_info.relative_mesh_file_name, input_file_path);
  auto mesh           = serac::buildMeshFromFile(full_mesh_path, mesh_info.ser_ref_levels, mesh_info.par_ref_levels);

  // Define the solid solver object
  auto                  solid_solver_info = (*inlet)["nonlinear_solid"].get<serac::NonlinearSolid::InputInfo>();
  serac::NonlinearSolid solid_solver(mesh, solid_solver_info);

  // Project the initial and reference configuration functions onto the
  // appropriate grid functions
  int dim = mesh->Dimension();

  mfem::VectorFunctionCoefficient defo_coef(dim, serac::initialDeformation);

  mfem::Vector velo(dim);
  velo = 0.0;

  mfem::VectorConstantCoefficient velo_coef(velo);

  // initialize x_cur, boundary condition, deformation, and
  // incremental nodal displacment grid functions by projection the
  // VectorFunctionCoefficient function onto them
  solid_solver.setDisplacement(defo_coef);
  solid_solver.setVelocity(velo_coef);

  std::set<int> ess_bdr = {1};

  // define the displacement vector
  mfem::Vector disp(dim);
  disp           = 0.0;
  auto disp_coef = std::make_shared<mfem::VectorConstantCoefficient>(disp);

  std::set<int> trac_bdr = {2};

  // loading parameters
  // define the traction vector
  auto traction      = (*inlet)["nonlinear_solid/traction"].get<mfem::Vector>();
  auto traction_coef = std::make_shared<serac::VectorScaledConstantCoefficient>(traction);

  // Set the boundary condition information
  solid_solver.setDisplacementBCs(ess_bdr, disp_coef);
  solid_solver.setTractionBCs(trac_bdr, traction_coef);

  // Set the time step method
  solid_solver.setTimestepper(serac::TimestepMethod::QuasiStatic);

  // Complete the solver setup
  solid_solver.completeSetup();

  // initialize/set the time
  double t       = 0;
  double t_final = (*inlet)["t_final"];  // has default value
  double dt      = (*inlet)["dt"];       // has default value

  bool last_step = false;

  solid_solver.initializeOutput(serac::OutputType::VisIt, "serac");

  // enter the time step loop. This was modeled after example 10p.
  for (int ti = 1; !last_step; ti++) {
    double dt_real = std::min(dt, t_final - t);
    // compute current time
    t = t + dt_real;

    if (rank == 0) {
      std::cout << "step " << ti << ", t = " << t << std::endl;
    }

    traction_coef->SetScale(t);

    // Solve the Newton system
    solid_solver.advanceTimestep(dt_real);

    solid_solver.outputState();

    last_step = (t >= t_final - 1e-8 * dt);
  }

  serac::exitGracefully();
}

// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file without_input_file.cpp
 *
 * @brief A simple example of steady-state liquid crystal elastomer on that uses
 * the C++ API to configure the simulation
 */

#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/terminator.hpp"

#include "liquid_crystal_elastomer_functional.hpp"
#include "liquid_crystal_elastomer_functional_material.hpp"
#include "parameterized_liquid_crystal_elastomer_functional_material.hpp"

#include <fstream>

#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"

// _main_init_start
int main(int argc, char* argv[])
{
  /*auto [num_procs, rank] = */serac::initialize(argc, argv);

  int serial_refinement   = 1;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "LCE_functional_static_solve");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/star.mesh";
      // (dim == 2) ? SERAC_REPO_DIR "/data/meshes/star.mesh" : SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";

  auto mesh = serac::mesh::refineAndDistribute(serac::buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  // const int dim = mesh->SpaceDimension();
  constexpr int dim = 2; 
  constexpr int p   = 1;

  if(dim != 2 && dim != 3) 
  {
    std::cout<<"Dimension must be 2 or 3 for thermal functional test"<<std::endl; 
    exit(0);
  }

  // Define a boundary attribute set
  std::set<int> ess_bdr = {1};

  // Construct a functional-based thermal conduction solver
  serac::ThermalConductionFunctional<p, dim> LCE_solver(serac::Thermal::defaultQuasistaticOptions(), "LCE_functional");

  serac::tensor<double, dim, dim> cond = {{{5.0, 0.01}, {0.01, 1.0}}};

  serac::Thermal::LinearConductor<dim> mat(1.0, 1.0, cond);
  LCE_solver.setMaterial(mat);

  // Define the function for the initial temperature and boundary condition
  auto one = [](const mfem::Vector&, double) -> double { return 1.0; };

  // Set the initial temperature and boundary condition
  LCE_solver.setTemperatureBCs(ess_bdr, one);
  LCE_solver.setTemperature(one);

  // Define a constant source term
  serac::Thermal::ConstantSource source{1.0};
  LCE_solver.setSource(source);

  // Set the flux term to zero for testing code paths
  serac::Thermal::ConstantFlux flux_bc{0.0};
  LCE_solver.setFluxBCs(flux_bc);

  // _output_type_start
  LCE_solver.initializeOutput(serac::OutputType::ParaView, "LCE_output");
  // _output_type_end

  // Finalize the data structures
  LCE_solver.completeSetup();

  // Perform the quasi-static solve
  double dt = 1.0;
  LCE_solver.advanceTimestep(dt);

  // Output the sidre-based plot files
  LCE_solver.outputState();

  // _exit_start
  serac::exitGracefully();
}
// _exit_end

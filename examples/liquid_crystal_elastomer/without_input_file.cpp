// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file without_input_file.cpp
 *
 * @brief A simple example of steady-state thermal conduction that uses
 * the C++ API to configure the simulation
 */

// _incl_thermal_header_start
// #include "serac/physics/thermal_conduction.hpp"
// _incl_thermal_header_end
// _incl_state_manager_start
// #include "serac/physics/state/state_manager.hpp"
// _incl_state_manager_end
// _incl_infra_start
#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/terminator.hpp"
// _incl_infra_end
// _incl_mesh_start
// #include "serac/mesh/mesh_utils.hpp"
// _incl_mesh_end

#include "liquid_crystal_elastomer_functional.hpp"
#include "liquid_crystal_elastomer_functional_material.hpp"
#include "parameterized_liquid_crystal_elastomer_functional_material.hpp"
// #include "serac/physics/thermal_conduction_functional.hpp"
// #include "serac/physics/materials/thermal_functional_material.hpp"
// #include "serac/physics/materials/parameterized_thermal_functional_material.hpp"

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
  serac::StateManager::initialize(datastore, "thermal_functional_static_solve");

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
  serac::ThermalConductionFunctional<p, dim> thermal_solver(serac::Thermal::defaultQuasistaticOptions(), "thermal_functional");

  serac::tensor<double, dim, dim> cond = {{{5.0, 0.01}, {0.01, 1.0}}};

  serac::Thermal::LinearConductor<dim> mat(1.0, 1.0, cond);
  thermal_solver.setMaterial(mat);

  // Define the function for the initial temperature and boundary condition
  auto one = [](const mfem::Vector&, double) -> double { return 1.0; };

  // Set the initial temperature and boundary condition
  thermal_solver.setTemperatureBCs(ess_bdr, one);
  thermal_solver.setTemperature(one);

  // Define a constant source term
  serac::Thermal::ConstantSource source{1.0};
  thermal_solver.setSource(source);

  // Set the flux term to zero for testing code paths
  serac::Thermal::ConstantFlux flux_bc{0.0};
  thermal_solver.setFluxBCs(flux_bc);

  // Finalize the data structures
  thermal_solver.completeSetup();

  // Perform the quasi-static solve
  double dt = 1.0;
  thermal_solver.advanceTimestep(dt);

  // Output the sidre-based plot files
  thermal_solver.outputState();

  // _exit_start
  serac::exitGracefully();
}
// _exit_end

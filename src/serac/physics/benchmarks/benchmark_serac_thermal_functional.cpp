// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/thermal_conduction_functional.hpp"
#include "serac/physics/materials/thermal_functional_material.hpp"

#include <fstream>

#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/infrastructure/profiling.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"

template <int p, int dim>
void functional_test_static()
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement   = 1;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "thermal_functional_static_solve");

  static_assert(dim == 2 || dim == 3, "Dimension must be 2 or 3 for thermal functional test");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename =
      (dim == 2) ? SERAC_REPO_DIR "/data/meshes/star.mesh" : SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  // Define a boundary attribute set
  std::set<int> ess_bdr = {1};

  // Construct a functional-based thermal conduction solver
  ThermalConductionFunctional<p, dim> thermal_solver(ThermalConductionFunctional<p, dim>::defaultQuasistaticOptions(),
                                                     "thermal_functional");

  tensor<double, dim, dim> cond;

  // Define an anisotropic conductor material model
  if constexpr (dim == 2) {
    cond = {{{5.0, 0.01}, {0.01, 1.0}}};
  }

  if constexpr (dim == 3) {
    cond = {{{1.5, 0.01, 0.0}, {0.01, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
  }

  Thermal::LinearConductor<dim> mat(1.0, 1.0, cond);
  thermal_solver.setMaterial(mat);

  // Define the function for the initial temperature and boundary condition
  auto one = [](const mfem::Vector&, double) -> double { return 1.0; };

  // Set the initial temperature and boundary condition
  thermal_solver.setTemperatureBCs(ess_bdr, one);
  thermal_solver.setTemperature(one);

  // Define a constant source term
  Thermal::ConstantSource source{1.0};
  thermal_solver.setSource(source);

  // Set the flux term to zero for testing code paths
  Thermal::FluxBoundary flux_bc{0.0};
  thermal_solver.setFluxBCs(flux_bc);

  // Finalize the data structures
  thermal_solver.completeSetup();

  // Perform the quasi-static solve
  double dt = 1.0;
  thermal_solver.advanceTimestep(dt);

  // Output the sidre-based plot files
  thermal_solver.outputState();
}

template <int p, int dim>
void functional_test_dynamic()
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement   = 1;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "thermal_functional_dynamic_solve");

  static_assert(dim == 2 || dim == 3, "Dimension must be 2 or 3 for thermal functional test");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename =
      (dim == 2) ? SERAC_REPO_DIR "/data/meshes/star.mesh" : SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  // Define a boundary attribute set
  std::set<int> ess_bdr = {1};

  // Construct a functional-based thermal conduction solver
  ThermalConductionFunctional<p, dim> thermal_solver(ThermalConductionFunctional<p, dim>::defaultDynamicOptions(),
                                                     "thermal_functional");

  // Define an isotropic conductor material model
  Thermal::LinearIsotropicConductor mat(1.0, 1.0, 1.0);

  thermal_solver.setMaterial(mat);

  // Define the function for the initial temperature and boundary condition
  auto initial_temp = [](const mfem::Vector& x, double) -> double {
    if (x[0] < 0.5 || x[1] < 0.5) {
      return 1.0;
    }
    return 0.0;
  };

  // Set the initial temperature and boundary condition
  thermal_solver.setTemperatureBCs(ess_bdr, initial_temp);
  thermal_solver.setTemperature(initial_temp);

  // Define a constant source term
  Thermal::ConstantSource source{1.0};
  thermal_solver.setSource(source);

  // Set the flux term to zero for testing code paths
  Thermal::FluxBoundary flux_bc{0.0};
  thermal_solver.setFluxBCs(flux_bc);

  // Finalize the data structures
  thermal_solver.completeSetup();

  // Perform the time stepping
  double dt = 0.25;

  for (int i = 0; i < 4; ++i) {
    thermal_solver.outputState();
    thermal_solver.advanceTimestep(dt);
  }

  // Output the sidre-based plot files
  thermal_solver.outputState();
}

//------------------------------------------------------------------------------
#include "axom/slic/core/SimpleLogger.hpp"

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  // Initialize profiling
  serac::profiling::initialize();

  // Add metadata
  SERAC_SET_METADATA("test", "thermal_functional");

  // Profile code
  SERAC_MARK_BEGIN("Thermal Functional");

  SERAC_MARK_BEGIN("2D Linear Static");
  functional_test_static<1, 2>();
  SERAC_MARK_END("2D Linear Static");

  SERAC_MARK_BEGIN("2D Quadratic Static");
  functional_test_static<2, 2>();
  SERAC_MARK_END("2D Quadratic Static");

  SERAC_MARK_BEGIN("3D Linear Static");
  functional_test_static<1, 3>();
  SERAC_MARK_END("3D Linear Static");

  SERAC_MARK_BEGIN("3D Quadratic Static");
  functional_test_static<2, 3>();
  SERAC_MARK_END("3D Quadratic Static");

  SERAC_MARK_BEGIN("2D Linear Dynamic");
  functional_test_dynamic<1, 2>();
  SERAC_MARK_END("2D Linear Dynamic");

  SERAC_MARK_BEGIN("2D Quadratic Dynamic");
  functional_test_dynamic<2, 2>();
  SERAC_MARK_END("2D Quadratic Dynamic");

  SERAC_MARK_BEGIN("3D Linear Dynamic");
  functional_test_dynamic<1, 3>();
  SERAC_MARK_END("3D Linear Dynamic");

  SERAC_MARK_BEGIN("3D Quadratic Dynamic");
  functional_test_dynamic<2, 3>();
  SERAC_MARK_END("3D Quadratic Dynamic");

  // Finalize profiling
  serac::profiling::finalize();

  MPI_Finalize();

  return 0;
}

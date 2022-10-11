// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/physics/coefficients/coefficient_extensions.hpp"
#include "serac/infrastructure/input.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/solid.hpp"
#include "serac/serac_config.hpp"
#include "test_utilities.hpp"
#include "serac/physics/solid_functional.hpp"
#include "serac/physics/materials/solid_functional_material.hpp"

namespace serac {

TEST(SolidSolver, BdrPartition)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement   = 4;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_functional_bdr");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/square.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  // Define a boundary attribute set
  std::set<int> ess_bdr = {1};

  // Use a direct solver (DSuperLU) for the Jacobian solve
  SolverOptions options = {DirectSolverOptions{}, solid_mechanics::default_nonlinear_options};

  // Construct a functional-based solid mechanics solver
  SolidFunctional<1, 2> solid_solver(options, GeometricNonlinearities::On, "solid_functional");

  solid_mechanics::NeoHookean mat{1.0, 1.0, 1.0};
  solid_solver.setMaterial(mat);

  // Define the function for the initial displacement and boundary condition
  auto bc = [](const mfem::Vector&, mfem::Vector& bc_vec) -> void { bc_vec = 0.0; };

  // Set the initial displacement and boundary condition
  solid_solver.setDisplacementBCs(ess_bdr, bc);
  solid_solver.setDisplacement(bc);

  tensor<double, 2> constant_force;

  constant_force[0] = 0.0;
  constant_force[1] = 5.0e-4;

  solid_mechanics::ConstantBodyForce<2> force{constant_force};
  solid_solver.addBodyForce(force);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  double dt = 1.0;
  solid_solver.advanceTimestep(dt);

  // Output the sidre-based and paraview plot files
  solid_solver.outputState("paraview_output");
}

}  // namespace serac

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}

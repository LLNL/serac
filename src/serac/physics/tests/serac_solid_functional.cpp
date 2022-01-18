// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_functional.hpp"

#include <fstream>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/solid_functional.hpp"
#include "serac/physics/materials/solid_functional_material.hpp"

namespace serac {

template <int p, int dim>
void functional_solid_test_static()
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement   = 1;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_functional_static_solve");

  static_assert(dim == 2 || dim == 3, "Dimension must be 2 or 3 for solid functional test");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename =
      (dim == 2) ? SERAC_REPO_DIR "/data/meshes/beam-quad.mesh" : SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  // Define a boundary attribute set
  std::set<int> ess_bdr = {1};

  // define the solver configurations
  const IterativeSolverOptions default_linear_options = {.rel_tol     = 1.0e-6,
                                                         .abs_tol     = 1.0e-10,
                                                         .print_level = 0,
                                                         .max_iter    = 500,
                                                         .lin_solver  = LinearSolver::GMRES,
                                                         .prec        = HypreBoomerAMGPrec{}};

  const NonlinearSolverOptions default_nonlinear_options = {
      .rel_tol = 1.0e-4, .abs_tol = 1.0e-8, .max_iter = 500, .print_level = 1};

  const typename SolidFunctional<p, dim>::SolverOptions default_static = {default_linear_options,
                                                                          default_nonlinear_options};

  // Construct a functional-based thermal conduction solver
  SolidFunctional<p, dim> solid_solver(default_static, GeometricNonlinearities::Off, FinalMeshOption::Reference,
                                       "solid_functional");

  Solid::LinearIsotropicElasticity<dim> mat(1.0, 1.0, 1.0);
  solid_solver.setMaterial(mat);

  // Define the function for the initial temperature and boundary condition
  auto zero = [](const mfem::Vector&, mfem::Vector& zero_vec) -> void { zero_vec = 0.0; };

  // Set the initial temperature and boundary condition
  solid_solver.setDisplacementBCs(ess_bdr, zero);
  solid_solver.setDisplacement(zero);

  tensor<double, dim> constant_force;

  constant_force[0] = 0.0;
  constant_force[1] = 0.1;

  if (dim == 3) {
    constant_force[2] = 0.0;
  }

  Solid::ConstantBodyForce<dim> force{constant_force};
  solid_solver.addBodyForce(force);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  double dt = 1.0;
  solid_solver.advanceTimestep(dt);

  // Output the sidre-based plot files
  solid_solver.outputState();

  // Check the final temperature norm
  // EXPECT_NEAR(expected_temp_norm, norm(solid_solver.temperature()), 1.0e-6);
}

TEST(solid_functional, 2D_linear_static) { functional_solid_test_static<1, 2>(); }
TEST(solid_functional, 2D_quad_static) { functional_solid_test_static<2, 2>(); }

TEST(solid_functional, 3D_linear_static) { functional_solid_test_static<1, 3>(); }
TEST(solid_functional, 3D_quad_static) { functional_solid_test_static<2, 3>(); }

}  // namespace serac

//------------------------------------------------------------------------------
#include "axom/slic/core/SimpleLogger.hpp"

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}

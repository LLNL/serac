// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_mechanics.hpp"

#include <functional>
#include <fstream>
#include <set>
#include <string>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/physics/materials/parameterized_solid_material.hpp"
#include "serac/serac_config.hpp"

using namespace serac;

void functional_solid_test_robin_condition()
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int p                   = 2;
  constexpr int dim                 = 3;
  int           serial_refinement   = 0;
  int           parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_mechanics_robin_condition_test");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";

  std::string mesh_tag{"mesh"};

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  // _solver_params_start
  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver  = NonlinearSolver::Newton,
                                                  .relative_tol   = 1.0e-12,
                                                  .absolute_tol   = 1.0e-12,
                                                  .max_iterations = 5000,
                                                  .print_level    = 1};

  SolidMechanics<p, dim> solid_solver(nonlinear_options, solid_mechanics::default_linear_options,
                                      solid_mechanics::default_quasistatic_options, GeometricNonlinearities::Off,
                                      "solid_mechanics", mesh_tag);
  // _solver_params_end

  solid_mechanics::LinearIsotropic mat{
      1.0,  // mass density
      1.0,  // bulk modulus
      1.0   // shear modulus
  };

  solid_solver.setMaterial(mat);

  // prescribe zero displacement in the y- and z-directions
  // at the supported end of the beam,
  std::set<int> support     = {1};
  auto          zero        = [](const mfem::Vector&) -> double { return 0.0; };
  int           y_direction = 1;
  int           z_direction = 2;
  solid_solver.setDisplacementBCs(support, zero, y_direction);
  solid_solver.setDisplacementBCs(support, zero, z_direction);

  // clang-format off
  solid_solver.addCustomBoundaryIntegral(DependsOn<>{}, 
      [](double /* t */, auto position, auto displacement, auto /*acceleration*/) {
        auto [X, dX_dxi] = position;
        auto [u, du_dxi] = displacement;
        auto f           = u * 3.0 * (X[0] < 0.01);
        return f;  // define a displacement-proportional traction at the support
      });
  // clang-format on

  // apply an axial displacement at the the tip of the beam
  auto translated_in_x = [](const mfem::Vector&, double t, mfem::Vector& u) -> void {
    u    = 0.0;
    u[0] = t;
  };
  std::set<int> tip = {2};
  solid_solver.setDisplacementBCs(tip, translated_in_x);

  auto zero_displacement = [](const mfem::Vector&, mfem::Vector& u) -> void { u = 0.0; };
  solid_solver.setDisplacement(zero_displacement);

  // Finalize the data structures
  solid_solver.completeSetup();

  solid_solver.outputStateToDisk("robin_condition");

  // Perform the quasi-static solve
  int    num_steps = 1;
  double tmax      = 1.0;
  double dt        = tmax / num_steps;
  for (int i = 0; i < num_steps; i++) {
    solid_solver.advanceTimestep(dt);
    solid_solver.outputStateToDisk("robin_condition");
  }
}

TEST(SolidMechanics, robin_condition) { functional_solid_test_robin_condition(); }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}

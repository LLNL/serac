// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/solid_functional.hpp"
#include "serac/physics/materials/solid_functional_material.hpp"
#include "serac/physics/materials/parameterized_solid_functional_material.hpp"
#include "serac/infrastructure/initialize.hpp"

namespace serac {

using solid_mechanics::default_static_options;

TEST(SolidFunctional, BoundaryCondition)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement   = 0;
  int parallel_refinement = 0;

  constexpr int dim = 2;
  constexpr int p   = 1;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_functional_boundary condition");

  std::string filename = SERAC_REPO_DIR "/data/meshes/beam-quad.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  // Construct a functional-based solid mechanics solver
  SolidFunctional<p, dim> solid_solver(default_static_options, GeometricNonlinearities::Off,
                                       "solid_functional_boundary");

  solid_mechanics::LinearIsotropic<dim> mat{1.0, 1.0, 1.0};
  solid_solver.setMaterial(mat);

  // note: L is specific to beam-quad.mesh
  double L   = 8.0;
  double u_0 = 0.1;

  // Define the function for the Dirichlet boundary conditions
  auto zero_bc      = [](const mfem::Vector&, mfem::Vector& bc_vec) -> void { bc_vec = 0.0; };
  auto displaced_bc = [=](const mfem::Vector&, mfem::Vector& bc_vec) -> void {
    bc_vec[0] = u_0;
    bc_vec[1] = 0.0;
  };

  // Set the initial displacement and boundary condition
  solid_solver.setDisplacementBCs({1}, zero_bc);
  solid_solver.setDisplacementBCs({2}, displaced_bc);
  solid_solver.setDisplacement(zero_bc);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  double dt = 0.1;

  solid_solver.advanceTimestep(dt);
  solid_solver.outputState("paraview_output");

  // compare approximate solution to the exact,
  // note: the y-component of displacement is zero because the elastic moduli
  // were chosen such that Poisson's ratio is identically zero
  mfem::VectorFunctionCoefficient exact_soln(dim, [=](auto x, auto& u) {
    u[0] = u_0 * (x[0] / L);
    u[1] = 0.0;
  });

  double L2error = solid_solver.displacement().gridFunction().ComputeLpError(2, exact_soln);
  EXPECT_NEAR(0.0, L2error, 1.0e-10);

  // sqrt(int_0^L (u_0 * (x/L))^2 dx) = u_0 * sqrt(L / 3.0)
  EXPECT_NEAR(u_0 * ::sqrt(L / 3.0), norm(solid_solver.displacement()), 1.0e-10);

  auto [num_ranks, my_rank] = getMPIInfo();

  // Ensure the boundary conditions are set properly
  if (my_rank == 1) {
    EXPECT_NEAR(0.0, solid_solver.displacement()(0), 1.0e-14);
    EXPECT_NEAR(0.0, solid_solver.displacement()(1), 1.0e-14);
    EXPECT_NEAR(0.0, solid_solver.displacement()(8), 1.0e-14);
    EXPECT_NEAR(0.0, solid_solver.displacement()(9), 1.0e-14);
  }

  if (my_rank == 0) {
    EXPECT_NEAR(0.1, solid_solver.displacement()(8), 1.0e-14);
    EXPECT_NEAR(0.0, solid_solver.displacement()(9), 1.0e-14);
    EXPECT_NEAR(0.1, solid_solver.displacement()(18), 1.0e-14);
    EXPECT_NEAR(0.0, solid_solver.displacement()(19), 1.0e-14);
  }
}

}  // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}

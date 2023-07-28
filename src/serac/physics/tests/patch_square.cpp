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
#include "serac/serac_config.hpp"

namespace serac {

TEST(Manufactured, TwoDimensional)
{
  constexpr int p   = 1;
  constexpr int dim = 2;

  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "patch_square_data");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/more_meshes/square_indbc.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), 3, 0);
  serac::StateManager::setMesh(std::move(mesh));

  serac::LinearSolverOptions linear_options{.linear_solver  = LinearSolver::GMRES,
                                            .preconditioner = Preconditioner::HypreAMG,
                                            .relative_tol   = 1.0e-6,
                                            .absolute_tol   = 1.0e-14,
                                            .max_iterations = 500,
                                            .print_level    = 1};


  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver  = NonlinearSolver::Newton,
                                                  .relative_tol   = 1.0e-9,
                                                  .absolute_tol   = 1.0e-12,
                                                  .max_iterations = 5000,
                                                  .print_level    = 1};

  SolidMechanics<p, dim> solid_solver(nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options,
                                      GeometricNonlinearities::On, "solid_mechanics");
  double E = 1e6;
  double nu = 0.25;
  double                             K = E/(3*(1-2*nu));
  double                             G = E/(2*(1+nu));
  solid_mechanics::StVenantKirchhoff mat{1.0, K, G};
  solid_solver.setMaterial(mat);

  // Define the function for the initial displacement and boundary condition
  // from parameterized_thermomechanics_example.cpp
  // set up essential boundary conditions
  std::set<int> x_equals_0 = {3};
  std::set<int> y_equals_0 = {1};
  std::set<int> x_disp = {4};
  std::set<int> y_disp = {2};

  auto zero_scalar = [](const mfem::Vector&) -> double { return 0.0; };
  solid_solver.setDisplacementBCs(x_equals_0, zero_scalar, 0);
  solid_solver.setDisplacementBCs(y_equals_0, zero_scalar, 1);

  //traction tensor
  const tensor<double, 3> t1{{660206*4.85, 0, 0}};
  const tensor<double, 3> t2{{0, 120412*3.3, 0}};
  auto traction = [t1, t2](const auto& x, const tensor<double, dim>&, const double) {
    const double spatial_tolerance = 1e-6;
    if (x[0] > 1.0 - spatial_tolerance) {
      return t1;
    } else if (x[1]> 1.0 - spatial_tolerance) {
      return t2;
    } else {
      return 0*t1;
    }
  };
  solid_solver.setPiolaTraction(traction);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  double dt = 1.0;
  solid_solver.advanceTimestep(dt);

  // Output the sidre-based plot files
  solid_solver.outputState("visit_output");


  auto exact_disp = [](const mfem::Vector& X, mfem::Vector& u) {
    // u = x - X, where x = 2*X + 0*Y + 0*Z
    u[0] = X[0];
    u[1] = 0;
    u[2] = 0;
  };

  // Compute norm of error
  mfem::VectorFunctionCoefficient exact_solution_coef(dim, exact_disp);
  double error = computeL2Error(solid_solver.displacement(), exact_solution_coef);
  EXPECT_LT(error, 1e-10);
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

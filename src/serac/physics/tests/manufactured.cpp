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
  serac::StateManager::initialize(datastore, "manufactured_data");

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
                                                  .relative_tol   = 1.0e-12,
                                                  .absolute_tol   = 1.0e-12,
                                                  .max_iterations = 5000,
                                                  .print_level    = 1};

  SolidMechanics<p, dim> solid_solver(nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options,
                                      GeometricNonlinearities::On, "solid_mechanics");

  double                             K = 1.91666666666667;
  double                             G = 1.0;
  solid_mechanics::StVenantKirchhoff mat{1.0, K, G};
  solid_solver.setMaterial(mat);

// COMMENT OUT BCS FROM EXAMPLE
  // Define the function for the initial displacement and boundary condition
  // auto bc = [](const mfem::Vector&, mfem::Vector& bc_vec) -> void { bc_vec = 0.0; };

  // Define a boundary attribute set and specify initial / boundary conditions
  // std::set<int> ess_bdr = {1}; 
  // solid_solver.setDisplacementBCs(ess_bdr, bc);
  // solid_solver.setDisplacement(bc);

// TEMPLATE OF BCS EQUATION FROM HPP
  // setDisplacementBCs(const std::set<int>& disp_bdr, 
  //                    std::function<double(const mfem::Vector& x)> disp,
  //                    int component)

//TRIAL BCS FOR 3 INPUT PARAMETERS
//from parameterized_thermomechanics_example.cpp
  // set up essential boundary conditions
  std::set<int> x_equals_0 = {1};
  std::set<int> y_equals_0 = {2};

  auto zero_scalar = [](const mfem::Vector&) -> double { return 0.0; };
  auto zero_scalar_x = [](const mfem::Vector&) -> double { return 1.0; };
  solid_solver.setDisplacementBCs(x_equals_0, zero_scalar_x, 0);
  solid_solver.setDisplacementBCs(y_equals_0, zero_scalar, 1);
//CONT w rest of example code

  solid_solver.setPiolaTraction(
      [](const auto& x, const tensor<double, dim>& n, const double) { return -0.01 * n * (x[1] > 0.99); });

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  double dt = 1.0;
  solid_solver.advanceTimestep(dt);

  // Output the sidre-based plot files
  //solid_solver.outputState("visit_output");
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

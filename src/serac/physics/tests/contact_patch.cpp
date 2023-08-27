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

TEST(ContactTest, patch)
{
  constexpr int p   = 1;
  constexpr int dim = 3;

  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "contact_patch_data");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/twohex_for_contact.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), 2, 0);
  serac::StateManager::setMesh(std::move(mesh));

  serac::LinearSolverOptions linear_options{.linear_solver = LinearSolver::SuperLU, .print_level = 1};

#ifdef MFEM_USE_SUNDIALS
  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver  = NonlinearSolver::KINFullStep,
                                                  .relative_tol   = 1.0e-12,
                                                  .absolute_tol   = 1.0e-12,
                                                  .max_iterations = 20,
                                                  .print_level    = 1};
#else
  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver  = NonlinearSolver::Newton,
                                                  .relative_tol   = 1.0e-12,
                                                  .absolute_tol   = 1.0e-12,
                                                  .max_iterations = 20,
                                                  .print_level    = 1};
#endif

  serac::ContactOptions contact_options{.method      = ContactMethod::SingleMortar,
                                        .enforcement = ContactEnforcement::LagrangeMultiplier,
                                        .type        = ContactType::Frictionless,
                                        .penalty     = 1.0e4};

  SolidMechanics<p, dim> solid_solver(nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options,
                                      GeometricNonlinearities::On, "solid_mechanics");

  double                      K = 10.0;
  double                      G = 0.25;
  solid_mechanics::NeoHookean mat{1.0, K, G};
  solid_solver.setMaterial(mat);

  // Define the function for the initial displacement and boundary condition
  auto zero_disp_bc    = [](const mfem::Vector&) { return 0.0; };
  auto nonzero_disp_bc = [](const mfem::Vector&) { return -0.01; };

  // Define a boundary attribute set and specify initial / boundary conditions
  solid_solver.setDisplacementBCs({1}, zero_disp_bc, 0);
  solid_solver.setDisplacementBCs({2}, zero_disp_bc, 1);
  solid_solver.setDisplacementBCs({3}, zero_disp_bc, 2);
  solid_solver.setDisplacementBCs({6}, nonzero_disp_bc, 2);

  solid_solver.addContactPair(0, {4}, {5}, contact_options);

  // Finalize the data structures
  solid_solver.completeSetup();

  solid_solver.outputState("paraview_output");

  // Perform the quasi-static solve
  double dt = 1.0;
  solid_solver.advanceTimestep(dt);

  // Output the sidre-based plot files
  solid_solver.outputState("paraview_output");
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

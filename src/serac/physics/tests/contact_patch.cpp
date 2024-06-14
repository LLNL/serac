// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_mechanics_contact.hpp"

#include <functional>
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

class ContactTest : public testing::TestWithParam<std::pair<ContactEnforcement, std::string>> {};

TEST_P(ContactTest, patch)
{
  // NOTE: p must be equal to 1 for now
  constexpr int p   = 1;
  constexpr int dim = 3;

  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  std::string            name = "contact_patch_" + GetParam().second;
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/twohex_for_contact.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), 2, 0);
  StateManager::setMesh(std::move(mesh), "patch_mesh");

  LinearSolverOptions linear_options{.linear_solver = LinearSolver::Strumpack, .print_level = 1};
#ifndef MFEM_USE_STRUMPACK
  SLIC_INFO_ROOT("Contact requires MFEM built with strumpack.");
  return;
#endif

  NonlinearSolverOptions nonlinear_options{.nonlin_solver  = NonlinearSolver::Newton,
                                           .relative_tol   = 1.0e-12,
                                           .absolute_tol   = 1.0e-12,
                                           .max_iterations = 200,
                                           .print_level    = 1};

  ContactOptions contact_options{.method      = ContactMethod::SingleMortar,
                                 .enforcement = GetParam().first,
                                 .type        = ContactType::Frictionless,
                                 .penalty     = 1.0e4};

  SolidMechanicsContact<p, dim> solid_solver(nonlinear_options, linear_options,
                                             solid_mechanics::default_quasistatic_options, GeometricNonlinearities::On,
                                             name, "patch_mesh");

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

  // Add the contact interaction
  solid_solver.addContactInteraction(0, {4}, {5}, contact_options);

  // Finalize the data structures
  solid_solver.completeSetup();

  // std::string paraview_name = name + "_paraview";
  // solid_solver.outputStateToDisk(paraview_name);

  // Perform the quasi-static solve
  double dt = 1.0;
  solid_solver.advanceTimestep(dt);

  // Output the sidre-based plot files
  // solid_solver.outputStateToDisk(paraview_name);

  // Check the l2 norm of the displacement dofs
  auto                            c = (3.0 * K - 2.0 * G) / (3.0 * K + G);
  mfem::VectorFunctionCoefficient elasticity_sol_coeff(3, [c](const mfem::Vector& x, mfem::Vector& u) {
    u[0] = 0.25 * 0.01 * c * x[0];
    u[1] = 0.25 * 0.01 * c * x[1];
    u[2] = -0.5 * 0.01 * x[2];
  });
  mfem::ParFiniteElementSpace     elasticity_fes(solid_solver.reactions().space());
  mfem::ParGridFunction           elasticity_sol(&elasticity_fes);
  elasticity_sol.ProjectCoefficient(elasticity_sol_coeff);
  mfem::ParGridFunction approx_error(elasticity_sol);
  approx_error -= solid_solver.displacement().gridFunction();
  auto approx_error_l2 = mfem::ParNormlp(approx_error, 2, MPI_COMM_WORLD);
  EXPECT_NEAR(0.0, approx_error_l2, 1.0e-3);
}

INSTANTIATE_TEST_SUITE_P(tribol, ContactTest,
                         testing::Values(std::make_pair(ContactEnforcement::Penalty, "penalty"),
                                         std::make_pair(ContactEnforcement::LagrangeMultiplier,
                                                        "lagrange_multiplier")));

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

// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_mechanics.hpp"

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

class ContactTest : public testing::TestWithParam<std::pair<ContactEnforcement, std::string>> {
};

TEST_P(ContactTest, beam)
{
  // NOTE: p must be equal to 1 for now
  constexpr int p   = 1;
  constexpr int dim = 3;

  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  std::string            name = "contact_beam_" + GetParam().second;
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex-with-contact-block.mesh";

  // NOTE: The number of MPI ranks must be <= the min number of elements on a
  // contact face until Tribol PR #23 is included in Serac's Tribol
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), 2, 0);
  StateManager::setMesh(std::move(mesh));

  LinearSolverOptions linear_options{.linear_solver = LinearSolver::SuperLU, .print_level = 1};

  // NOTE: kinsol does not appear to be working with penalty
  NonlinearSolverOptions nonlinear_options{.nonlin_solver  = NonlinearSolver::Newton,
                                           .relative_tol   = 1.0e-12,
                                           .absolute_tol   = 1.0e-12,
                                           .max_iterations = 200,
                                           .print_level    = 1};

  ContactOptions contact_options{.method      = ContactMethod::SingleMortar,
                                 .enforcement = GetParam().first,
                                 .type        = ContactType::TiedSlide,
                                 .penalty     = 1.0e4};

  SolidMechanics<p, dim> solid_solver(nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options,
                                      GeometricNonlinearities::On, name);

  double                      K = 10.0;
  double                      G = 0.25;
  solid_mechanics::NeoHookean mat{1.0, K, G};
  solid_solver.setMaterial(mat);

  // Pass the BC information to the solver object
  solid_solver.setDisplacementBCs({1}, [](const mfem::Vector&, mfem::Vector& u) {
    u.SetSize(dim);
    u = 0.0;
  });
  solid_solver.setDisplacementBCs({6}, [](const mfem::Vector&, mfem::Vector& u) {
    u.SetSize(dim);
    u    = 0.0;
    u[2] = -0.15;
  });

  // Add the contact interaction
  solid_solver.addContactInteraction(0, {7}, {5}, contact_options);

  // Finalize the data structures
  solid_solver.completeSetup();

  std::string paraview_name = name + "_paraview";
  solid_solver.outputState(paraview_name);

  // Perform the quasi-static solve
  double dt = 1.0;
  solid_solver.advanceTimestep(dt);

  // Output the sidre-based plot files
  solid_solver.outputState(paraview_name);

  // Check the l2 norm of the displacement dofs
  auto u_l2 = mfem::ParNormlp(solid_solver.displacement(), 2, MPI_COMM_WORLD);
  EXPECT_NEAR(3.3257055635785537, u_l2, 1.0e-3);
}

// NOTE: if Penalty is first and Lagrange Multiplier is second, super LU gives a
// zero diagonal error
INSTANTIATE_TEST_SUITE_P(tribol, ContactTest,
                         testing::Values(std::make_pair(ContactEnforcement::LagrangeMultiplier, "lagrange_multiplier"),
                                         std::make_pair(ContactEnforcement::Penalty, "penalty")));

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

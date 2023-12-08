// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
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
#include "serac/physics/materials/parameterized_solid_material.hpp"
#include "serac/serac_config.hpp"

#include "tribol/interface/mfem_tribol.hpp"
#include "redecomp/common/TypeDefs.hpp"

namespace serac {

class ContactTest : public testing::TestWithParam<std::tuple<ContactEnforcement, ContactType, std::string>> {};

TEST_P(ContactTest, sphere)
{
  // NOTE: p must be equal to 1 for now
  constexpr int p   = 1;
  constexpr int dim = 3;

  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  std::string            name = "contact_sphere_" + std::get<2>(GetParam());
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/twist_sphere.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), 0, 0);
  StateManager::setMesh(std::move(mesh), "sphere_mesh");

  LinearSolverOptions linear_options{.linear_solver = LinearSolver::SuperLU, .print_level = 1};

  // NOTE: kinsol does not appear to be working with penalty
  NonlinearSolverOptions nonlinear_options{.nonlin_solver  = NonlinearSolver::Newton,
                                           .relative_tol   = 1.0e-8,
                                           .absolute_tol   = 1.0e-5,
                                           .max_iterations = 200,
                                           .print_level    = 1};

  ContactOptions contact_options{.method      = ContactMethod::SingleMortar,
                                 .enforcement = std::get<0>(GetParam()),
                                 .type        = std::get<1>(GetParam()),
                                 .penalty     = 1.0e5};

  SolidMechanicsContact<p, dim, Parameters<L2<0>, L2<0>>> solid_solver(
      nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options, GeometricNonlinearities::On,
      name, "sphere_mesh");

  FiniteElementState       K_field(StateManager::newState(L2<0, 3>{}, "bulk_mod", "ironing_mesh"));
  mfem::Vector             K_values({10.0, 100.0});
  mfem::PWConstCoefficient K_coeff(K_values);
  K_field.project(K_coeff);
  solid_solver.setParameter(0, K_field);

  FiniteElementState       G_field(StateManager::newState(L2<0, 3>{}, "shear_mod", "ironing_mesh"));
  mfem::Vector             G_values({0.25, 2.5});
  mfem::PWConstCoefficient G_coeff(G_values);
  G_field.project(G_coeff);
  solid_solver.setParameter(1, G_field);

  solid_mechanics::ParameterizedNeoHookeanSolid<dim> mat{1.0, 0.0, 0.0};
  solid_solver.setMaterial(DependsOn<0, 1>{}, mat);

  // Pass the BC information to the solver object
  solid_solver.setDisplacementBCs({5}, [](const mfem::Vector&, mfem::Vector& u) {
    u.SetSize(dim);
    u = 0.0;
  });
  double time = 0.0;
  solid_solver.setDisplacementBCs({10}, [&time](const mfem::Vector& x, mfem::Vector& u) {
    u.SetSize(dim);
    u = 0.0;
    if (time <= 3.0 + 1.0e-12) {
      u[2] = -time * 0.02;
    } else {
      u[0] = (std::cos(redecomp::pi / 40.0 * (time - 3.0)) - 1.0) * (x[0] - 0.5) -
             std::sin(redecomp::pi / 40.0 * (time - 3.0)) * (x[1] - 0.5);
      u[1] = std::sin(redecomp::pi / 40.0 * (time - 3.0)) * (x[0] - 0.5) +
             (std::cos(redecomp::pi / 40.0 * (time - 3.0)) - 1.0) * (x[1] - 0.5);
      u[2] = -0.06;
    }
  });

  // Add the contact interaction
  solid_solver.addContactInteraction(0, {6}, {7}, contact_options);

  // Finalize the data structures
  solid_solver.completeSetup();

  std::string paraview_name = name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);

  tribol::saveRedecompMesh(0);

  // Perform the quasi-static solve
  double dt = 1.0;

  for (int i{0}; i < 23; ++i) {
    time += dt;

    solid_solver.advanceTimestep(dt);

    // Output the sidre-based plot files
    solid_solver.outputStateToDisk(paraview_name);

    tribol::saveRedecompMesh(i + 1);
  }

  // Check the l2 norm of the displacement dofs
  // auto u_l2 = mfem::ParNormlp(solid_solver.displacement(), 2, MPI_COMM_WORLD);
  // if (std::get<1>(GetParam()) == ContactType::TiedNormal) {
  //   EXPECT_NEAR(3.3257055635785537, u_l2, 2.0e-2);
  // } else if (std::get<1>(GetParam()) == ContactType::Frictionless) {
  //   EXPECT_NEAR(3.4771738496372739, u_l2, 1.0e-2);
  // }
}

// NOTE: if Penalty is first and Lagrange Multiplier is second, super LU gives a
// zero diagonal error
INSTANTIATE_TEST_SUITE_P(tribol, ContactTest,
                         testing::Values(std::make_tuple(ContactEnforcement::Penalty, ContactType::Frictionless,
                                                         "penalty_frictionless")));

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

// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>

#include <string>

#include "axom/slic/core/SimpleLogger.hpp"

#include "mfem.hpp"

#include "serac/physics/solid_mechanics_contact.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/parameterized_solid_material.hpp"
#include "serac/serac_config.hpp"

#include "tribol/interface/mfem_tribol.hpp"

int main(int argc, char* argv[])
{
  serac::initialize(argc, argv);
  
  // NOTE: p must be equal to 1
  constexpr int p   = 1;
  // NOTE: dim must be equal to 3
  constexpr int dim = 3;

  // Create DataStore
  std::string            name = "contact_sphere_example";
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/twist_sphere.mesh";

  auto mesh = serac::mesh::refineAndDistribute(serac::buildMeshFromFile(filename), 0, 0);
  serac::StateManager::setMesh(std::move(mesh), "sphere_mesh");

  serac::LinearSolverOptions linear_options{.linear_solver = serac::LinearSolver::Strumpack, .print_level = 1};
#ifndef MFEM_USE_STRUMPACK
  SLIC_INFO_ROOT("Contact requires MFEM built with strumpack.");
  return;
#endif

  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver  = serac::NonlinearSolver::Newton,
                                           .relative_tol   = 1.0e-8,
                                           .absolute_tol   = 1.0e-5,
                                           .max_iterations = 200,
                                           .print_level    = 1};

  serac::ContactOptions contact_options{.method      = serac::ContactMethod::SingleMortar,
                                 .enforcement = serac::ContactEnforcement::Penalty,
                                 .type        = serac::ContactType::Frictionless,
                                 .penalty     = 1.0e5};

  serac::SolidMechanicsContact<p, dim, serac::Parameters<serac::L2<0>, serac::L2<0>>> solid_solver(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options, serac::GeometricNonlinearities::On,
      name, "sphere_mesh", {"bulk_mod", "shear_mod"});

  serac::FiniteElementState       K_field(serac::StateManager::newState(serac::L2<0>{}, "bulk_mod", "sphere_mesh"));
  mfem::Vector             K_values({10.0, 100.0});
  mfem::PWConstCoefficient K_coeff(K_values);
  K_field.project(K_coeff);
  solid_solver.setParameter(0, K_field);

  serac::FiniteElementState       G_field(serac::StateManager::newState(serac::L2<0>{}, "shear_mod", "sphere_mesh"));
  mfem::Vector             G_values({0.25, 2.5});
  mfem::PWConstCoefficient G_coeff(G_values);
  G_field.project(G_coeff);
  solid_solver.setParameter(1, G_field);

  serac::solid_mechanics::ParameterizedNeoHookeanSolid<dim> mat{1.0, 0.0, 0.0};
  solid_solver.setMaterial(serac::DependsOn<0, 1>{}, mat);

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
      u[0] = (std::cos(M_PI / 40.0 * (time - 3.0)) - 1.0) * (x[0] - 0.5) -
             std::sin(M_PI / 40.0 * (time - 3.0)) * (x[1] - 0.5);
      u[1] = std::sin(M_PI / 40.0 * (time - 3.0)) * (x[0] - 0.5) +
             (std::cos(M_PI / 40.0 * (time - 3.0)) - 1.0) * (x[1] - 0.5);
      u[2] = -0.06;
    }
  });

  // Add the contact interaction
  solid_solver.addContactInteraction(0, {6}, {7}, contact_options);

  // Finalize the data structures
  solid_solver.completeSetup();

  std::string paraview_name = name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);

  // Perform the quasi-static solve
  double dt = 1.0;

  for (int i{0}; i < 23; ++i) {
    time += dt;

    solid_solver.advanceTimestep(dt);

    // Output the sidre-based plot files
    solid_solver.outputStateToDisk(paraview_name);
  }
}
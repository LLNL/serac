// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <set>
#include <string>

#include "axom/slic/core/SimpleLogger.hpp"
#include "axom/inlet.hpp"

#include "mfem.hpp"

#include "serac/physics/solid_mechanics_contact.hpp"
#include "serac/infrastructure/terminator.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/parameterized_solid_material.hpp"
#include "serac/serac_config.hpp"

using namespace serac;

// std::function<std::string(const std::string&)> petscPCTypeValidator = [](const std::string& in) -> std::string {
//   return std::to_string(static_cast<int>(mfem_ext::stringToPetscPCType(in)));
// };

int main(int argc, char* argv[])
{
  constexpr int dim = 3;
  constexpr int p   = 1;

  // Command line arguments
  int  serial_refinement   = 0;
  int  parallel_refinement = 0;
  bool use_contact         = false;
  auto contact_type        = serac::ContactEnforcement::LagrangeMultiplier;
  int  num_steps           = 20;

  NonlinearSolverOptions nonlinear_options     = solid_mechanics::default_nonlinear_options;
  LinearSolverOptions    linear_options        = solid_mechanics::default_linear_options;
  nonlinear_options.nonlin_solver              = serac::NonlinearSolver::TrustRegion;
  nonlinear_options.relative_tol               = 1e-8;
  nonlinear_options.absolute_tol               = 1e-12;
  nonlinear_options.min_iterations             = 1;
  nonlinear_options.max_iterations             = 500;
  nonlinear_options.max_line_search_iterations = 20;
  nonlinear_options.print_level                = 1;
  nonlinear_options.force_monolithic           = true;
#ifdef SERAC_USE_PETSC
  // linear_options.linear_solver        = serac::LinearSolver::PetscGMRES;
  // linear_options.preconditioner       = serac::Preconditioner::Petsc;
  // linear_options.petsc_preconditioner = serac::PetscPCType::HMG;
  // linear_options.relative_tol   = 1e-8;
  // linear_options.absolute_tol   = 1e-16;
  // linear_options.max_iterations = 10000;
#endif

  // Contact specific options
  double penalty = 1e3;

  serac::initialize(argc, argv);
  axom::slic::setLoggingMsgLevel(axom::slic::message::Level::Debug);

  // Handle command line arguments
  axom::CLI::App app{"Hollow cylinder buckling example"};
  app.add_option("--serial-refinement", serial_refinement, "Serial refinement steps", true);
  app.add_option("--parallel-refinement", parallel_refinement, "Parallel refinement steps", true);
  app.add_option("--nonlinear-solver", nonlinear_options.nonlin_solver, "Nonlinear solver", true);
  app.add_option("--linear-solver", linear_options.linear_solver, "Linear solver", true);
  app.add_option("--preconditioner", linear_options.preconditioner, "Preconditioner", true);
  // app.add_option("--petsc-pc-type", linear_options.petsc_preconditioner, "Petsc preconditioner", true)
  //     ->transform(
  //         [](const std::string& in) -> std::string {
  //           return std::to_string(static_cast<int>(mfem_ext::stringToPetscPCType(in)));
  //         },
  //         "Convert string to PetscPCType", "PetscPCTypeTransform");
  app.add_option("--num-steps", num_steps, "Number of pseudo-time step", true);
  app.add_flag("--contact", use_contact, "Use contact for the inner faces of the cylinder");
  app.add_option("--contact-type", contact_type,
                 "Type of contact enforcement, 0 for penalty or 1 for Lagrange multipliers", true)
      ->needs("--contact");
  app.add_option("--penalty", penalty, "Penalty for contact", true)->needs("--contact");
  app.set_help_flag("--help");
  app.allow_extras()->parse(argc, argv);

  // nonlinear_options.force_monolithic = linear_options.preconditioner != Preconditioner::Petsc;

  // Create DataStore
  std::string            name     = use_contact ? "buckling_cylinder_contact" : "buckling_cylinder";
  std::string            mesh_tag = "mesh";
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, name + "_data");
  std::string filename = SERAC_REPO_DIR "/data/meshes/hollow-cylinder.mesh";

  auto mesh  = serac::buildMeshFromFile(filename);
  auto pmesh = mesh::refineAndDistribute(std::move(mesh), serial_refinement, parallel_refinement);

  serac::StateManager::setMesh(std::move(pmesh), mesh_tag);

  std::unique_ptr<SolidMechanics<p, dim>> solid_solver;
  if (use_contact) {
    auto solid_contact_solver = std::make_unique<serac::SolidMechanicsContact<p, dim>>(
        nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options,
        serac::GeometricNonlinearities::On, name, mesh_tag);

    // Add the contact interaction
    serac::ContactOptions contact_options{.method      = serac::ContactMethod::SingleMortar,
                                          .enforcement = contact_type,
                                          .type        = serac::ContactType::Frictionless,
                                          .penalty     = penalty};
    auto                  contact_interaction_id = 100;
    std::set<int>         xneg{2};
    std::set<int>         xpos{3};
    solid_contact_solver->addContactInteraction(contact_interaction_id, xpos, xneg, contact_options);
    solid_solver = std::move(solid_contact_solver);
  } else {
    solid_solver = std::make_unique<serac::SolidMechanics<p, dim>>(nonlinear_options, linear_options,
                                                                   serac::solid_mechanics::default_quasistatic_options,
                                                                   serac::GeometricNonlinearities::On, name, mesh_tag);
  }

  // Define a Neo-Hookean material
  auto                        lambda = 1.0;
  auto                        G      = 0.1;
  solid_mechanics::NeoHookean mat{.density = 1.0, .K = (3 * lambda + 2 * G) / 3, .G = G};

  solid_solver->setMaterial(mat);

  // set up essential boundary conditions
  std::set<int> bottom = {1};
  std::set<int> top    = {4};
  auto          clamp  = [](const mfem::Vector&, mfem::Vector& u) {
    u.SetSize(dim);
    u = 0.0;
  };
  auto compress = [&](const mfem::Vector&, double t, mfem::Vector& u) {
    u.SetSize(dim);
    u    = 0.0;
    u(0) = u(2) = -1.5 / std::sqrt(2.0) * t / num_steps;
  };
  solid_solver->setDisplacementBCs(bottom, clamp);
  solid_solver->setDisplacementBCs(top, compress);

  solid_solver->completeSetup();
  std::string paraview_name = name + "_paraview";
  solid_solver->outputStateToDisk(paraview_name);

  // Perform the quasi-static solve
  SLIC_INFO_ROOT(axom::fmt::format("Running hollow cylinder bucking example with {} displacement dofs",
                                   solid_solver->displacement().GlobalSize()));
  SLIC_INFO_ROOT("Starting pseudo-timestepping.");
  serac::logger::flush();
  for (int i = 0; i <= num_steps; i++) {
    SLIC_INFO_ROOT(axom::fmt::format("time = {} (out of {})", solid_solver->time(), num_steps));
    serac::logger::flush();
    solid_solver->advanceTimestep(1.0);

    // Output the sidre-based plot files
    solid_solver->outputStateToDisk(paraview_name);
  }
  SLIC_INFO_ROOT(axom::fmt::format("final time = {}", solid_solver->time()));

  serac::exitGracefully();
}

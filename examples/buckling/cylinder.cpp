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

mfem::Array<int> arrayFromSet(std::set<int> orig)
{
  auto array = mfem::Array<int>();
  array.Reserve(static_cast<int>(orig.size()));
  for (const auto& val : orig) {
    array.Append(val);
  }
  return array;
}

// std::function<std::string(const std::string&)> petscPCTypeValidator = [](const std::string& in) -> std::string {
//   return std::to_string(static_cast<int>(mfem_ext::stringToPetscPCType(in)));
// };

int main(int argc, char* argv[])
{
  constexpr int dim = 3;
  constexpr int p   = 1;

  // Command line arguments
  int    serial_refinement   = 0;
  int    parallel_refinement = 0;
  bool   use_contact         = false;
  auto   contact_type        = serac::ContactEnforcement::LagrangeMultiplier;
  double dt                  = 0.1;

  NonlinearSolverOptions nonlinear_options     = solid_mechanics::default_nonlinear_options;
  LinearSolverOptions    linear_options        = solid_mechanics::default_linear_options;
  nonlinear_options.nonlin_solver              = serac::NonlinearSolver::TrustRegion;
  nonlinear_options.relative_tol               = 1e-6;
  nonlinear_options.absolute_tol               = 1e-10;
  nonlinear_options.min_iterations             = 1;
  nonlinear_options.max_iterations             = 500;
  nonlinear_options.max_line_search_iterations = 20;
  nonlinear_options.print_level                = 1;
#ifdef SERAC_USE_PETSC
  linear_options.linear_solver        = serac::LinearSolver::GMRES;
  linear_options.preconditioner       = serac::Preconditioner::Petsc;
  linear_options.petsc_preconditioner = serac::PetscPCType::HMG;
  linear_options.preconditioner       = serac::Preconditioner::HypreAMG;
  linear_options.relative_tol         = 1e-8;
  linear_options.absolute_tol         = 1e-16;
  linear_options.max_iterations       = 10000;
  // linear_options.preconditioner_print_level = 1;
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
  app.add_option("--dt", dt, "Size of pseudo-time step pre-contact", true);
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

  std::set<int> xneg{2};
  std::set<int> xpos{3};
  std::set<int> bottom   = {1};
  std::set<int> top      = {4};
  std::string   filename = SERAC_REPO_DIR "/data/meshes/hollow-cylinder.mesh";

  // std::set<int> xneg{3};
  // std::set<int> xpos{4};
  // std::set<int> bottom   = {2};
  // std::set<int> top      = {1};
  // std::string   filename = SERAC_REPO_DIR "/data/meshes/cyl.g";

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
    auto                  contact_interaction_id = 0;
    solid_contact_solver->addContactInteraction(contact_interaction_id, xneg, xpos, contact_options);
    solid_solver = std::move(solid_contact_solver);
  } else {
    solid_solver = std::make_unique<serac::SolidMechanics<p, dim>>(nonlinear_options, linear_options,
                                                                   serac::solid_mechanics::default_quasistatic_options,
                                                                   serac::GeometricNonlinearities::On, name, mesh_tag);
    auto domain  = serac::Domain::ofBoundaryElements(
         StateManager::mesh(mesh_tag), [&](std::vector<vec3>, int attr) { return xpos.find(attr) != xpos.end(); });
    solid_solver->setPressure(
        [&](auto&, double t) {
          auto p = 0.01 * t;
          std::cout << axom::fmt::format("applied pressure {}\n", p);
          return p;
        },
        domain);
  }

  // Define a Neo-Hookean material
  auto                        lambda = 1.0;
  auto                        G      = 0.1;
  solid_mechanics::NeoHookean mat{.density = 1.0, .K = (3 * lambda + 2 * G) / 3, .G = G};

  solid_solver->setMaterial(mat);

  // set up essential boundary conditions

  auto clamp = [](const mfem::Vector&, mfem::Vector& u) {
    u.SetSize(dim);
    u = 0.0;
  };
  auto compress = [&](const mfem::Vector&, double t, mfem::Vector& u) {
    u.SetSize(dim);
    u    = 0.0;
    u(0) = u(2) = -1.5 / std::sqrt(2.0) * t;
  };
  solid_solver->setDisplacementBCs(bottom, clamp);
  solid_solver->setDisplacementBCs(top, compress);

  // Finalize the data structures
  solid_solver->completeSetup();
  auto& tribol_config = tribol::parameters_t::getInstance();
  // tribol_config.auto_interpen_check = true;
  tribol_config.output_directory = name + "_tribol";
  axom::utilities::filesystem::makeDirsForPath(tribol_config.output_directory);
  tribol_config.vis_cycle_incr = 1;
  tribol_config.vis_type       = tribol::VIS_MESH_FACES_AND_OVERLAPS;

  std::string paraview_name = name + "_paraview";
  solid_solver->outputStateToDisk(paraview_name);

  // Perform the quasi-static solve
  SLIC_INFO_ROOT(axom::fmt::format("Running hollow cylinder bucking example with {} displacement dofs",
                                   solid_solver->displacement().GlobalSize()));
  SLIC_INFO_ROOT("Starting pseudo-timestepping.");
  serac::logger::flush();
  while (solid_solver->time() < 1.0 && std::abs(solid_solver->time() - 1) > DBL_EPSILON) {
    SLIC_INFO_ROOT(axom::fmt::format("time = {} (out of 1.0)", solid_solver->time()));
    serac::logger::flush();

    auto next_dt = solid_solver->time() < 0.65 ? dt : dt * 0.1;
    solid_solver->advanceTimestep(next_dt);

    // Output the sidre-based plot files
    solid_solver->outputStateToDisk(paraview_name);
  }
  SLIC_INFO_ROOT(axom::fmt::format("final time = {}", solid_solver->time()));

  serac::exitGracefully();
}

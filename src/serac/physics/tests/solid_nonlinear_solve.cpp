// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_mechanics.hpp"
#include "serac/physics/solid_mechanics_contact.hpp"

#include <functional>
#include <fstream>
#include <set>
#include <string>
#include <algorithm>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include <serac/physics/materials/liquid_crystal_elastomer.hpp>

#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/serac_config.hpp"
#include "serac/infrastructure/profiling.hpp"
#include "serac/infrastructure/terminator.hpp"

//#include "mesh/vtk.hpp"
//#include <vtkSimplePointsWriter.h>
//#include <vtkSphereSource.h>

#include "petscmat.h"

using namespace serac;

std::string mesh_path = ".";

enum Prec
{
  JACOBI,
  STRUMPACK,
  CHOLESKI,
  LU,
  MULTIGRID
};

enum NonlinSolve
{
  NEWTON,
  LINESEARCH,
  CRITICALPOINT,
  TRUSTREGION
};

NonlinSolve nonlinSolve = NonlinSolve::TRUSTREGION;
Prec prec = Prec::JACOBI;


auto get_opts(int max_iters, double abs_tol = 1e-9)
{
  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::TrustRegion,
                                                  //.nonlin_solver = NonlinearSolver::NewtonLineSearch, //LineSearch,
                                                  //.nonlin_solver = NonlinearSolver::PetscNewton,  //LineSearch,
                                                  //.nonlin_solver = NonlinearSolver::PetscNewtonCriticalPoint, // breaks for snap_cell
                                                  .relative_tol               = abs_tol,
                                                  .absolute_tol               = abs_tol,
                                                  .min_iterations             = 1,
                                                  .max_iterations             = 2000,
                                                  .max_line_search_iterations = 15,
                                                  .print_level                = 1};

  // best for critical point newton: ls = PetscGMRES, petsc_preconditioner = PetscPCType::LU;
  serac::LinearSolverOptions linear_options = {.linear_solver  = LinearSolver::CG,
                                               //.linear_solver  = LinearSolver::PetscGMRES,
                                               //.linear_solver  = LinearSolver::PetscCG,
                                               .preconditioner = Preconditioner::HypreJacobi,
                                               //.preconditioner = Preconditioner::Petsc,
                                               //.petsc_preconditioner = PetscPCType::JACOBI,
                                               //.petsc_preconditioner = PetscPCType::JACOBI_ROWMAX,
                                               //.petsc_preconditioner = PetscPCType::GAMG, 
                                               //.petsc_preconditioner = PetscPCType::HMG, // Zach's prefered
                                               //.petsc_preconditioner = PetscPCType::LU,
                                               //.petsc_preconditioner = PetscPCType::CHOLESKY,
                                               .relative_tol   = 0.7 * abs_tol,
                                               .absolute_tol   = 0.7 * abs_tol,
                                               .max_iterations = max_iters,
                                               .print_level = 0};

  switch(nonlinSolve) {
  case NonlinSolve::NEWTON: {
    printf("using newton solver\n");
    nonlinear_options.min_iterations=0;
    nonlinear_options.max_line_search_iterations=0;
    nonlinear_options.nonlin_solver = NonlinearSolver::Newton;
    break;
  }
  case NonlinSolve::LINESEARCH: {
    printf("using newton linesearch solver\n");
    nonlinear_options.min_iterations=0;
    nonlinear_options.nonlin_solver = NonlinearSolver::PetscNewtonBacktracking;
    //nonlinear_options.nonlin_solver = NonlinearSolver::NewtonLineSearch;
    break;
  }
  case NonlinSolve::CRITICALPOINT: {
    printf("using newton critical point solver\n");
    nonlinear_options.min_iterations=0;
    nonlinear_options.max_line_search_iterations=0;
    nonlinear_options.nonlin_solver = NonlinearSolver::PetscNewtonCriticalPoint;
    break;
  }
  case NonlinSolve::TRUSTREGION: {
    printf("using trust region solver\n");
    nonlinear_options.nonlin_solver = NonlinearSolver::TrustRegion;
    break;
  }
  }

  switch (prec) {
  case Prec::JACOBI: {
    printf("using jacobi\n");
    linear_options.linear_solver = LinearSolver::CG;
    linear_options.preconditioner = Preconditioner::HypreJacobi;
    break;
  }
  case Prec::STRUMPACK: {
    printf("using strumpack\n");
    linear_options.linear_solver = LinearSolver::Strumpack;
    break;
  }
  case Prec::CHOLESKI: {
    printf("using choleski\n");
    //linear_options.linear_solver = LinearSolver::GMRES;
    linear_options.linear_solver = LinearSolver::CG;
    linear_options.preconditioner = Preconditioner::Petsc;
    linear_options.petsc_preconditioner = PetscPCType::CHOLESKY;
    break;
  }
  case Prec::LU: {
    printf("using lu\n");
    linear_options.linear_solver = LinearSolver::GMRES;
    linear_options.preconditioner = Preconditioner::Petsc;
    linear_options.petsc_preconditioner = PetscPCType::LU;
    break;
  }
  case Prec::MULTIGRID: {
    printf("using multigrid\n");
    linear_options.linear_solver = LinearSolver::GMRES;
    linear_options.preconditioner = Preconditioner::Petsc;
    linear_options.petsc_preconditioner = PetscPCType::HMG;
    break;
  }
  default: {
    printf("error\n");
    exit(1);
  }
  }

  return std::make_pair(nonlinear_options, linear_options);
}

#include <ostream>
#include <fstream>

void write_sphere(std::vector<std::array<double,3>> coords, std::vector<double> rad, int step)
{
  std::stringstream file; // = "sphere_";
  file << "sphere_" << std::setfill('0') << std::setw(2) << std::to_string(step) << ".vtk";
  std::ofstream fout(file.str());

  fout << "# vtk DataFile Version 3.0\n";
  fout << "this may work\n";
  fout << "ASCII\n";
  fout << "DATASET UNSTRUCTURED_GRID\n";

  fout << "POINTS " + std::to_string(coords.size()) + " double\n";
  for (size_t i=0; i < coords.size(); ++i) {
    fout << std::to_string(coords[i][0]) + " " + std::to_string(coords[i][1]) + " " + std::to_string(coords[i][2]) + "\n";
  }
  fout << "POINT_DATA " + std::to_string(coords.size()) + "\n";
  fout << "SCALARS sphere_radius double\n";
  fout << "LOOKUP_TABLE default\n";
  for (size_t i=0; i < rad.size(); ++i) {
    fout << std::to_string(rad[i]) + "\n";
  }
}

void functional_solid_test_buckle_ball()
{
  // initialize serac
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "buckleBallStore");

  static constexpr int ORDER{1};
  static constexpr int DIM{3};

  //int Nx = 6;
  //int Ny = 9;
  //int Nz = 13*5;

  int Nx = 5;
  int Ny = 8;
  int Nz = 14*5;

  double Lx = 1.0;
  double Ly = 6.0;
  double Lz = 30.0;

  double density       = 1.0;
  double E             = 1000.0;
  double v             = 0.33;
  double bulkMod       = E / (3. * (1. - 2. * v));
  double shearMod      = E / (2. * (1. + v));
  double loadMagnitude = 2.0;
  double v_crit = 5e-2; //2.0;
  //double eta = 0.45;
  //double mu = 0.3;
  double mu = 0.01;

  std::string    meshTag = "mesh";
  mfem::Mesh     mesh    = mfem::Mesh::MakeCartesian3D(Nx, Ny, Nz, mfem::Element::HEXAHEDRON, Lx, Ly, Lz);
  auto           pmesh   = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, mesh);
  mfem::ParMesh* meshPtr = &serac::StateManager::setMesh(std::move(pmesh), meshTag);

  // solid mechanics
  using seracSolidType = serac::SolidMechanics<ORDER, DIM, serac::Parameters<>>;

  auto [nonlinear_options, linear_options] = get_opts(Nx*Ny*Nz, 1e-9);

  auto seracSolid = std::make_unique<seracSolidType>(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options,
      serac::GeometricNonlinearities::On, "serac_solid", meshTag, std::vector<std::string>{});

  double radius = 5.0;
  double initial_penalty = 1000.0;
  std::array<double,DIM> rigid_velo_0{0.0, -0.0, 2.0};
  std::array<double,DIM> corner_0{-11.5, 3.0, -3.5};
  auto lset_0 = std::make_unique<LevelSetSphere<DIM>>(corner_0, radius, rigid_velo_0);
  auto friction_0 = std::make_unique<NodalFriction<DIM>>(mu, v_crit, rigid_velo_0);
  auto constraint_0 = std::make_unique<InequalityConstraint<ORDER, DIM>>(std::move(lset_0), std::move(friction_0),
                                                                         "serac_solid_"+std::to_string(0), meshTag,
                                                                         initial_penalty);
  seracSolid->addInequalityConstraint(std::move(constraint_0));

  int num_time_steps = 22;
  double total_time = num_time_steps;
  double dt = total_time / num_time_steps;

  double t_meet = (Lz/2.0 - corner_0[2]) / rigid_velo_0[2] - dt;
  std::array<double,DIM> rigid_velo_1{2.0, 0.0, -0.2};
  std::array<double,DIM> corner_1; for (size_t i=0; i < DIM; ++i) corner_1[i] = corner_0[i] + rigid_velo_0[i] * t_meet - rigid_velo_1[i] * t_meet;
  auto lset_1 = std::make_unique<LevelSetSphere<DIM>>(corner_1, radius, rigid_velo_1);
  auto friction_1 = std::make_unique<NodalFriction<DIM>>(mu, v_crit, rigid_velo_1);
  auto constraint_1 = std::make_unique<InequalityConstraint<ORDER, DIM>>(std::move(lset_1), std::move(friction_1),
                                                                         "serac_solid_"+std::to_string(1), meshTag,
                                                                         initial_penalty);
  seracSolid->addInequalityConstraint(std::move(constraint_1));

  // at 12.0, want v = (3,0,0)
  // x2 + v2 * 12 = x1 + v1 * 12

  serac::solid_mechanics::NeoHookean material{density, bulkMod, shearMod};
  seracSolid->setMaterial(serac::DependsOn<>{}, material);

  serac::Domain backSurface = serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(3)); // 4,5 with traction makes a twist
  serac::Domain topSurface = serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(6));

  //seracSolid->setPressure([&](auto, auto t) { return t > 0 ? loadMagnitude : 0.0; }, backSurface);
  seracSolid->setPressure([&](auto, auto t) { return t > 0.5 * dt && t < 1.5 * dt ? loadMagnitude : -1e-5 * loadMagnitude; }, backSurface);
  // seracSolid->setTraction([&](auto, auto n, auto t) { return  t > 0.5 * dt && t < 1.5 * dt ? -loadMagnitude * n : 1e-7 * loadMagnitude * n; }, backSurface);

  // displacement on bottom surface
  seracSolid->setDisplacementBCs({1}, [](const mfem::Vector&, mfem::Vector& u) {
    u = 0.0;
  });

  // fix displacement on top surface
  seracSolid->setDisplacementBCs({6}, [](const mfem::Vector&, mfem::Vector& u) {
    u = 0.0;
    u[2] = -10.0;
  });

  auto getSphere = [=](int num, const double time) {
    std::array<double,DIM> sphere_x;
    if (num==0) {
      for (size_t i=0; i < DIM; ++i) sphere_x[i] = corner_0[i] + rigid_velo_0[i] * time;
    } else {
      for (size_t i=0; i < DIM; ++i) sphere_x[i] = corner_1[i] + rigid_velo_1[i] * time;
    }
    return sphere_x;
  };

  seracSolid->completeSetup();

  double time = 0.0;
  seracSolid->outputStateToDisk("paraview_buckle_ball");
  write_sphere( {getSphere(0,time), getSphere(1,time) }, {radius, radius}, 0 );
  seracSolid->advanceTimestep(0.2);
  time += 0.2;

  seracSolid->outputStateToDisk("paraview_buckle_ball");
  write_sphere( {getSphere(0,time), getSphere(1,time) }, {radius, radius}, 1 );
  for (int step=0; step < num_time_steps; ++step) {
    seracSolid->advanceTimestep(dt);
    time += dt;

    seracSolid->outputStateToDisk("paraview_buckle_ball");
    write_sphere( {getSphere(0,time), getSphere(1,time) }, {radius, radius}, step+2 );
  }
}

void functional_solid_test_euler()
{
  // initialize serac
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "eulerStore");

  static constexpr int ORDER{1};
  static constexpr int DIM{3};

  int Nx = 5;
  int Ny = 8;
  int Nz = 15*5;
  //int Nx = 4;
  //int Ny = 7;
  //int Nz = 10*5;

  double Lx = 1.0;
  double Ly = 2.0;
  double Lz = 30.0;

  double density       = 1.0;
  double E             = 10.0;
  double v             = 0.33;
  double bulkMod       = E / (3. * (1. - 2. * v));
  double shearMod      = E / (2. * (1. + v));
  //double eta = 0.45;

  std::string    meshTag = "mesh";
  mfem::Mesh     mesh    = mfem::Mesh::MakeCartesian3D(Nx, Ny, Nz, mfem::Element::HEXAHEDRON, Lx, Ly, Lz);
  auto           pmesh   = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, mesh);
  mfem::ParMesh* meshPtr = &serac::StateManager::setMesh(std::move(pmesh), meshTag);

  // solid mechanics
  using seracSolidType = serac::SolidMechanics<ORDER, DIM, serac::Parameters<>>;

  auto [nonlinear_options, linear_options] = get_opts(Nx*Ny*Nz, 1e-9);

  auto seracSolid = std::make_unique<seracSolidType>(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options,
      serac::GeometricNonlinearities::On, "serac_solid", meshTag, std::vector<std::string>{});

  serac::solid_mechanics::NeoHookean material{density, bulkMod, shearMod};
  seracSolid->setMaterial(serac::DependsOn<>{}, material);

  serac::Domain backSurface = serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(3)); // 4,5 with traction makes a twist
  serac::Domain topSurface = serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(6));

  int num_time_steps = 2;
  double total_time = num_time_steps;
  double dt = total_time / num_time_steps;

  //seracSolid->setPressure([&](auto, auto t) { return t > 0 ? loadMagnitude : 0.0; }, backSurface);
  //seracSolid->setPressure([&](auto, auto t) { return 0.014 * t; }, topSurface);
  seracSolid->setTraction([&](auto, auto n, auto t) { return -0.004 * t * n; }, topSurface);
  seracSolid->setTraction([&](auto, auto n, auto) { return 1e-6 * n; }, backSurface);

  // displacement on bottom surface
  seracSolid->setDisplacementBCs({1}, [](const mfem::Vector&, mfem::Vector& u) {
    u = 0.0;
  });

  seracSolid->completeSetup();

  seracSolid->outputStateToDisk("paraview_euler");
  for (int i=0; i < num_time_steps; ++i) {
    seracSolid->advanceTimestep(dt);
    seracSolid->outputStateToDisk("paraview_euler");
  }

}


void functional_solid_test_gate()
{
  // initialize serac
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "gateStore");

  static constexpr int ORDER{2};
  static constexpr int DIM{2};

  double density       = 1.0;
  double E             = 1.0;
  double v             = 0.33;
  double bulkMod       = E / (3. * (1. - 2. * v));
  double shearMod      = E / (2. * (1. + v));
  //double eta = 0.45;

  std::string    mesh_tag = "mesh";
  std::string input_file_name = mesh_path + "gate.g";
  auto mesh  = serac::buildMeshFromFile(input_file_name);
  auto pmesh = mesh::refineAndDistribute(std::move(mesh), 0, 0);
  mfem::ParMesh& meshRef = serac::StateManager::setMesh(std::move(pmesh), mesh_tag);

  // solid mechanics
  using seracSolidType = serac::SolidMechanics<ORDER, DIM, serac::Parameters<>>;

  auto [nonlinear_options, linear_options] = get_opts(10000, 1e-10);

  auto seracSolid = std::make_unique<seracSolidType>(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options,
      serac::GeometricNonlinearities::On, "serac_solid", mesh_tag, std::vector<std::string>{});

  serac::solid_mechanics::NeoHookean material{density, bulkMod, shearMod};
  seracSolid->setMaterial(serac::DependsOn<>{}, material);

  //serac::Domain symSurface = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<DIM>(1)); // 4,5 with traction makes a twist
  serac::Domain topSurface = serac::Domain::ofBoundaryElements(meshRef, serac::by_attr<DIM>(4));

  int num_time_steps = 12;
  double total_time = num_time_steps;
  double dt = total_time / num_time_steps;

  //seracSolid->setPressure([&](auto, auto t) { return t > 0 ? loadMagnitude : 0.0; }, backSurface);
  //seracSolid->setPressure([&](auto, auto) { return 12*loadMagnitude; }, topSurface);
  //seracSolid->setTraction([&](auto, auto n, auto) { return -0.5 * n; }, topSurface);

  seracSolid->setTraction([&](auto, auto n, auto time) { return time > 1.1 ? -1.0e-3 * time * n : 0.0 * n; }, topSurface);

  seracSolid->setDisplacementBCs({1}, [](const mfem::Vector&, mfem::Vector& u) {
    u = 0.0;
    u[0] = 15;
  });

  seracSolid->setDisplacementBCs({2}, [](const mfem::Vector&) {
    return 0.0;
  }, 0);

  seracSolid->completeSetup();

  seracSolid->outputStateToDisk("paraview_gate");
  for (int i=0; i < num_time_steps; ++i) {
    seracSolid->advanceTimestep(dt);
    seracSolid->outputStateToDisk("paraview_gate");
  }

}

void functional_solid_test_cylinder()
{
  constexpr int dim = 3;
  constexpr int p   = 1;

  // Command line arguments
  int  serial_refinement   = 0;
  int  parallel_refinement = 0;
  bool use_contact         = false;
  auto contact_type        = serac::ContactEnforcement::LagrangeMultiplier;
  int  num_steps           = 1;

  auto [nonlinear_options, linear_options] = get_opts(10000, 1e-9);

  // Contact specific options
  double penalty = 1e3;

  // Handle command line arguments
  // axom::CLI::App app{"Hollow cylinder buckling example"};
  // app.add_option("--serial-refinement", serial_refinement, "Serial refinement steps", true);
  // app.add_option("--parallel-refinement", parallel_refinement, "Parallel refinement steps", true);
  // app.add_option("--nonlinear-solver", nonlinear_options.nonlin_solver, "Nonlinear solver", true);
  // app.add_option("--linear-solver", linear_options.linear_solver, "Linear solver", true);
  // app.add_option("--preconditioner", linear_options.preconditioner, "Preconditioner", true);

  // app.add_option("--num-steps", num_steps, "Number of pseudo-time step", true);
  // app.add_flag("--contact", use_contact, "Use contact for the inner faces of the cylinder");
  // app.add_option("--contact-type", contact_type,
  //               "Type of contact enforcement, 0 for penalty or 1 for Lagrange multipliers", true)
  //     ->needs("--contact");
  // app.add_option("--penalty", penalty, "Penalty for contact", true)->needs("--contact");
  // app.set_help_flag("--help");
  // app.allow_extras()->parse(argc, argv);

  // nonlinear_options.force_monolithic = linear_options.preconditioner != Preconditioner::Petsc;

  // Create DataStore
  std::string            name     = use_contact ? "buckle_cylinder_contact" : "buckle_cylinder";
  std::string            mesh_tag = "mesh";
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, name + "Store");
  std::string input_file_name = mesh_path + "cyl.g";

  auto mesh  = serac::buildMeshFromFile(input_file_name);
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
    std::set<int>         xneg{3};
    std::set<int>         xpos{4};
    solid_contact_solver->addContactInteraction(contact_interaction_id, xpos, xneg, contact_options);
    solid_solver = std::move(solid_contact_solver);
  } else {
    solid_solver = std::make_unique<serac::SolidMechanics<p, dim>>(nonlinear_options, linear_options,
                                                                   serac::solid_mechanics::default_quasistatic_options,
                                                                   serac::GeometricNonlinearities::On, name, mesh_tag);
  }

  // Define a Neo-Hookean material
  auto                        lambda = 1.0;
  auto                        G      = 0.2;
  solid_mechanics::NeoHookean mat{.density = 1.0, .K = (3 * lambda + 2 * G) / 3, .G = G};

  solid_solver->setMaterial(mat);

  // set up essential boundary conditions
  std::set<int> bottom = {2};
  std::set<int> top    = {1};
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

  //seracSolid->setTraction([&](auto, auto n, auto t) { return  t > 0.5 * dt && t < 1.5 * dt ? -loadMagnitude * n : 1e-7 * loadMagnitude * n; }, backSurface);

  solid_solver->completeSetup();
  std::string paraview_name = "paraview_" + name;
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
}

#define UNBLOCK_THINGS
#ifdef UNBLOCK_THINGS

void functional_solid_test_nonlinear_buckle(double loadMagnitude)
{
  // initialize serac
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "buckleStore");

  static constexpr int ORDER{1};
  static constexpr int DIM{3};

  int Nx = 100;
  int Ny = 50;
  int Nz = 3;

  double Lx = 20.0;
  double Ly = 10.0;
  double Lz = 0.3;

  double density       = 1.0;
  double E             = 1.0;
  double v             = 0.33;
  double bulkMod       = E / (3. * (1. - 2. * v));
  double shearMod      = E / (2. * (1. + v));
  //double loadMagnitude = 1e-4; //2e-2; //0.2e-5;  // 2e-2;

  std::string    meshTag = "mesh";
  mfem::Mesh     mesh    = mfem::Mesh::MakeCartesian3D(Nx, Ny, Nz, mfem::Element::HEXAHEDRON, Lx, Ly, Lz);
  auto           pmesh   = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, mesh);
  mfem::ParMesh* meshPtr = &serac::StateManager::setMesh(std::move(pmesh), meshTag);

  // solid mechanics
  using seracSolidType = serac::SolidMechanics<ORDER, DIM, serac::Parameters<>>;

  auto [nonlinear_options, linear_options] = get_opts(3*Nx*Ny*Nz, 1e-8);

  auto seracSolid = std::make_unique<seracSolidType>(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options,
      serac::GeometricNonlinearities::On, "serac_solid", meshTag, std::vector<std::string>{});

  serac::solid_mechanics::NeoHookean material{density, bulkMod, shearMod};
  seracSolid->setMaterial(serac::DependsOn<>{}, material);

  // fix displacement on side surface
  seracSolid->setDisplacementBCs({2, 3, 4, 5}, [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; });

  serac::Domain topSurface = serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(6));
  //seracSolid->setTraction([&](auto, auto n, auto) { return -loadMagnitude * n; }, topSurface);
  seracSolid->setPressure([&](auto, auto) { return loadMagnitude; }, topSurface);
  seracSolid->completeSetup();
  seracSolid->advanceTimestep(1.0);

  seracSolid->outputStateToDisk("paraview_buckle_easy");
}


void functional_solid_test_friction_box()
{
  // initialize serac
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "boxStore");

  static constexpr int ORDER{1};
  static constexpr int DIM{3};

  int Nx = 5;
  int Ny = 5;
  int Nz = 5;

  double Lx = 5.0;
  double Ly = 5.0;
  double Lz = 8.3;

  double density       = 1.0;
  double E             = 1000.0;
  double v             = 0.33;
  double bulkMod       = E / (3. * (1. - 2. * v));
  double shearMod      = E / (2. * (1. + v));
  // double loadMagnitude = 1e-5; //0.2e-5;  // 2e-2;
  // double eta = 0.45;
  double mu = 0.3; //3;

  std::string    meshTag = "mesh";
  mfem::Mesh     mesh    = mfem::Mesh::MakeCartesian3D(Nx, Ny, Nz, mfem::Element::HEXAHEDRON, Lx, Ly, Lz);
  auto           pmesh   = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, mesh);
  mfem::ParMesh* meshPtr = &serac::StateManager::setMesh(std::move(pmesh), meshTag);

  // solid mechanics
  using seracSolidType = serac::SolidMechanics<ORDER, DIM, serac::Parameters<>>;

  auto [nonlinear_options, linear_options] = get_opts(Nx*Ny*Nz, 1e-9);

  auto seracSolid = std::make_unique<seracSolidType>(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options,
      serac::GeometricNonlinearities::On, "serac_solid", meshTag, std::vector<std::string>{});

  double initial_penalty = 1000.0;
  std::array<double,DIM> rigid_velo{0.0, 0.0, 0.0};
  std::array<double,DIM> corner{-0.1, -0.1, -0.1};
  std::array<double,DIM> plane_normal;
  for (size_t i=0; i < DIM; ++i) {
    plane_normal = {};
    plane_normal[i] = 1.0;
    auto lset = std::make_unique<LevelSetPlane<DIM>>(corner, plane_normal);
    auto friction = std::make_unique<NodalFriction<DIM>>(mu, 1e-2, rigid_velo);
    auto constraint = std::make_unique<InequalityConstraint<ORDER, DIM>>(std::move(lset), std::move(friction),
                                                                         "serac_solid_"+std::to_string(i), meshTag,
                                                                         initial_penalty);
    seracSolid->addInequalityConstraint(std::move(constraint));
  }

  serac::solid_mechanics::NeoHookean material{density, bulkMod, shearMod};
  seracSolid->setMaterial(serac::DependsOn<>{}, material);

  serac::Domain topSurface = serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(6));

  int num_time_steps = 1;
  double total_time = 1.0; //num_time_steps; //.0;
  double dt = total_time / num_time_steps;

  //const tensor<double, DIM> nx{1.0, 0.0, 0.0};
  //const tensor<double, DIM> ny{0.0, 1.0, 0.0};
  //const tensor<double, DIM> nz{0.0, 0.0, 1.0};
  //seracSolid->setTraction([&](auto, auto, double t) {
  //  //printf("t = %g\n", t);
  //  auto sideLoad = eta * (t / total_time) * loadMagnitude;
  //  return -loadMagnitude * ny - sideLoad / std::sqrt(2.0) * nx - sideLoad / std::sqrt(2.0) * nz;
  //}, topSurface);

  // fix displacement on top surface
  seracSolid->setDisplacementBCs({6}, [](const mfem::Vector&, mfem::Vector& u) {
    u = 0.0;
    u[2] = -0.15;
    u[1] = -0.05;
  });

  seracSolid->completeSetup();

  seracSolid->outputStateToDisk("paraview_friction_box");
  //seracSolid->advanceTimestep(0.0);
  //seracSolid->outputStateToDisk("paraview_friction_box");
  //seracSolid->advanceTimestep(0.0);
  //seracSolid->outputStateToDisk("paraview_friction_box");
  for (int i=0; i < num_time_steps; ++i) {
    seracSolid->advanceTimestep(dt);
    seracSolid->outputStateToDisk("paraview_friction_box");
  }
}


void functional_solid_test_nonlinear_arch()
{
  // initialize serac
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "archStore");

  static constexpr int ORDER{1};
  static constexpr int DIM{3};

  double density       = 1.0;
  double E             = 1.0;
  double v             = 0.33;
  double bulkMod       = E / (3. * (1. - 2. * v));
  double shearMod      = E / (2. * (1. + v));
  //double loadMagnitude = 1.2e-2; //0.2e-5;  // 2e-2;

  std::string meshTag = "mesh";
  std::string input_file_name = mesh_path + "arch.g";

  auto initial_mesh = serac::buildMeshFromFile(input_file_name);
  auto           pmesh   = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, initial_mesh);
  mfem::ParMesh* meshPtr = &serac::StateManager::setMesh(std::move(pmesh), meshTag);

  // solid mechanics
  using seracSolidType = serac::SolidMechanics<ORDER, DIM, serac::Parameters<>>;

  auto [nonlinear_options, linear_options] = get_opts(50000, 1e-8);

  auto seracSolid = std::make_unique<seracSolidType>(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options,
      serac::GeometricNonlinearities::On, "serac_solid", meshTag, std::vector<std::string>{});

  //auto lset = std::make_unique<LevelSetPlane<DIM>>(std::array<double,DIM>{0.0, -4.0, 0.0}, 
  //                                                 std::array<double,DIM>{0.0, 1.0, 0.0});
  double initial_penalty = 0.1;
  std::array<double,DIM> ball_velo{10.0, 0.0, 0.0};
  auto lset = std::make_unique<LevelSetSphere<DIM>>(std::array<double,DIM>{-5.0, -5.0, 1.5}, 2.2, ball_velo);
  auto friction = std::make_unique<NodalFriction<DIM>>(0.3, 1e-1, ball_velo);
  auto constraint = std::make_unique<InequalityConstraint<ORDER, DIM>>(std::move(lset), std::move(friction),
                                                                       "serac_solid", meshTag, initial_penalty);
  seracSolid->addInequalityConstraint(std::move(constraint));

  serac::solid_mechanics::NeoHookean material{density, bulkMod, shearMod};
  seracSolid->setMaterial(serac::DependsOn<>{}, material);

  // fix displacement on side surface
  seracSolid->setDisplacementBCs({2, 3}, [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; });

  serac::Domain topSurface = serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(1));
  //seracSolid->setTraction([&](auto, auto n, auto) { return -loadMagnitude * n; }, topSurface);
  //seracSolid->setPressure([&](auto, auto) { return loadMagnitude; }, topSurface);

  seracSolid->completeSetup();

  int num_steps = 16;

  seracSolid->outputStateToDisk("paraview_arch");
  for (int step=0; step < num_steps; ++step) {
    seracSolid->advanceTimestep(1.0 / num_steps);
    std::cout << "outputting at step " << step+1 << std::endl;
    seracSolid->outputStateToDisk("paraview_arch");
  }
  
}


void functional_solid_test_nonlinear_snap_cell()
{
  // initialize serac
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "snapCellStore");

  static constexpr int ORDER{2};
  static constexpr int DIM{2};

  double density       = 1.0;
  double E             = 1.0;
  double v             = 0.49;
  double bulkMod       = E / (3. * (1. - 2. * v));
  double shearMod      = E / (2. * (1. + v));
  double loadMagnitude = 4.e-2;

  std::string meshTag = "mesh";
  std::string input_file_name = mesh_path + "snap_cell.exo";

  auto initial_mesh = serac::buildMeshFromFile(input_file_name);
  auto           pmesh   = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, initial_mesh);
  mfem::ParMesh* meshPtr = &serac::StateManager::setMesh(std::move(pmesh), meshTag);

  // solid mechanics
  using seracSolidType = serac::SolidMechanics<ORDER, DIM, serac::Parameters<>>;

  auto [nonlinear_options, linear_options] = get_opts(50000, 1e-8);

  auto seracSolid = std::make_unique<seracSolidType>(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options,
      serac::GeometricNonlinearities::On, "serac_solid", meshTag, std::vector<std::string>{});

  serac::solid_mechanics::NeoHookean material{density, bulkMod, shearMod};
  seracSolid->setMaterial(serac::DependsOn<>{}, material);

  // fix displacement on side surface
  seracSolid->setDisplacementBCs({1}, [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; });
  seracSolid->setDisplacementBCs({2,3,5,6}, [](const mfem::Vector&) { return 0.0; }, 0);

  serac::Domain topSurface = serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(7));
  seracSolid->setPressure([&](auto, auto time) { return time * loadMagnitude; }, topSurface);
  // seracSolid->setTraction([&](auto, auto n, auto) { return -loadMagnitude * n; }, topSurface);

  seracSolid->completeSetup();

  seracSolid->outputStateToDisk("paraview_snap_cell");
  int num_steps = 10;
  for (int i=0; i < num_steps; ++i) {
    seracSolid->advanceTimestep(1.0 / num_steps);
    std::cout << "outputing at step " << i << "/" << num_steps << std::endl;
    seracSolid->outputStateToDisk("paraview_snap_cell");
  }
}


void functional_solid_test_nonlinear_snap_chain()
{
  // initialize serac
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "snapChainStore");

  static constexpr int ORDER{1};
  static constexpr int DIM{2};

  double density       = 1.0;
  double E             = 1.0;
  double v             = 0.49;
  double bulkMod       = E / (3. * (1. - 2. * v));
  double shearMod      = E / (2. * (1. + v));
  double loadMagnitude = 1.2e-2;

  std::string meshTag = "mesh";
  std::string input_file_name = mesh_path + "snap_chain.exo";

  auto initial_mesh = serac::buildMeshFromFile(input_file_name);
  auto           pmesh   = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, initial_mesh);
  mfem::ParMesh* meshPtr = &serac::StateManager::setMesh(std::move(pmesh), meshTag);

  // solid mechanics
  using seracSolidType = serac::SolidMechanics<ORDER, DIM, serac::Parameters<>>;

  auto [nonlinear_options, linear_options] = get_opts(50000, 1e-8);

  auto seracSolid = std::make_unique<seracSolidType>(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options,
      serac::GeometricNonlinearities::On, "serac_solid", meshTag, std::vector<std::string>{});

  serac::solid_mechanics::NeoHookean material{density, bulkMod, shearMod};
  seracSolid->setMaterial(serac::DependsOn<>{}, material);

  // fix displacement on side surface
  seracSolid->setDisplacementBCs({1}, [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; });
  seracSolid->setDisplacementBCs({2,4}, [](const mfem::Vector&) { return 0.0; }, 0);

  serac::Domain topSurface = serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(5));
  seracSolid->setPressure([&](auto, auto time) { return time * loadMagnitude; }, topSurface);
  // seracSolid->setTraction([&](auto, auto n, auto) { return -loadMagnitude * n; }, topSurface);

  seracSolid->completeSetup();

  seracSolid->outputStateToDisk("paraview_snap_chain");
  int num_steps = 10;
  for (int i=0; i < num_steps; ++i) {
    seracSolid->advanceTimestep(1.0 / num_steps);
    seracSolid->outputStateToDisk("paraview_snap_chain");
  }
}

TEST(SolidMechanics, nonlinear_solve_buckle_easy) { functional_solid_test_nonlinear_buckle(2e-5); }
TEST(SolidMechanics, nonlinear_solve_buckle_medium) { functional_solid_test_nonlinear_buckle(4e-4); }
TEST(SolidMechanics, nonlinear_solve_buckle_hard) { functional_solid_test_nonlinear_buckle(3e-2); }
TEST(SolidMechanics, nonlinear_solve_arch) { functional_solid_test_nonlinear_arch(); }
TEST(SolidMechanics, nonlinear_solve_snap_chain) { functional_solid_test_nonlinear_snap_chain(); }
TEST(SolidMechanics, nonlinear_solve_snap_cell) { functional_solid_test_nonlinear_snap_cell(); }
TEST(SolidMechanics, nonlinear_solve_friction_box) { 
  functional_solid_test_friction_box();
}
TEST(SolidMechanics, nonlinear_solid_test_gate) { functional_solid_test_gate(); }

#endif

TEST(SolidMechanics, nonlinear_solve_buckle_ball) { functional_solid_test_buckle_ball(); }
TEST(SolidMechanics, nonlinear_solve_euler) { functional_solid_test_euler(); }
TEST(SolidMechanics, nonlinear_solve_cylinder) { functional_solid_test_cylinder(); }

class InputParser
{
public:
  InputParser(int& argc, char** argv){
    for (int i=1; i < argc; ++i) {
      this->tokens.push_back(std::string(argv[i]));
    }
  }
  std::string getCmdOption(const std::string& option) const {
    std::vector<std::string>::const_iterator itr;
    itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
    if (itr != this->tokens.end() && ++itr != this->tokens.end()) {
        return *itr;
    }
    static const std::string empty_string("");
    return empty_string;
  }
  bool cmdOptionExists(const std::string& option) const {
    return std::find(this->tokens.begin(), this->tokens.end(), option)
            != this->tokens.end();
  }
private:
  std::vector <std::string> tokens;
};

int main(int argc, char* argv[])
{
  InputParser parser(argc, argv);
  auto filename = parser.getCmdOption("-p");
  if (!filename.empty()) {
    mesh_path = filename;
  }

  axom::CLI::App app{"Nonlinear problems"};
  app.add_option("--nonlinear-solver", nonlinSolve, "Nonlinear solver", true);
  app.add_option("--preconditioner", prec, "Preconditioner", true);
  app.set_help_flag("--help");
  app.allow_extras()->parse(argc, argv);

  ::testing::InitGoogleTest(&argc, argv);

  serac::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  serac::exitGracefully(result);

  return result;
}

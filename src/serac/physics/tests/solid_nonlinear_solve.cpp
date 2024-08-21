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

#include "serac/physics/materials/liquid_crystal_elastomer.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/serac_config.hpp"
#include "serac/infrastructure/profiling.hpp"
#include "serac/infrastructure/terminator.hpp"

using namespace serac;

std::string mesh_path = ".";

enum Prec
{
  JACOBI,
  STRUMPACK,
  CHOLESKI,
  LU,
  MULTIGRID,
  PETSC_MULTIGRID
};

enum NonlinSolve
{
  NEWTON,
  LINESEARCH,
  CRITICALPOINT,
  TRUSTREGION
};

NonlinSolve nonlinSolve = NonlinSolve::TRUSTREGION;
Prec        prec        = Prec::JACOBI;

auto get_opts(int max_iters, double abs_tol = 1e-9)
{
  serac::NonlinearSolverOptions nonlinear_options{
      .nonlin_solver = NonlinearSolver::TrustRegion,
      //.nonlin_solver = NonlinearSolver::NewtonLineSearch, //LineSearch,
      //.nonlin_solver = NonlinearSolver::PetscNewton,  //LineSearch,
      //.nonlin_solver = NonlinearSolver::PetscNewtonCriticalPoint, // breaks for snap_cell
      .relative_tol               = abs_tol,
      .absolute_tol               = abs_tol,
      .min_iterations             = 1,
      .max_iterations             = 2000,
      .max_line_search_iterations = 20,
      .print_level                = 1};

  // best for critical point newton: ls = PetscGMRES, petsc_preconditioner = PetscPCType::LU;
  serac::LinearSolverOptions linear_options = {.linear_solver = LinearSolver::CG,
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
                                               .print_level    = 1};

  switch (nonlinSolve) {
    case NonlinSolve::NEWTON: {
      SLIC_INFO_ROOT("using newton solver");
      nonlinear_options.min_iterations             = 0;
      nonlinear_options.max_line_search_iterations = 0;
      nonlinear_options.nonlin_solver              = NonlinearSolver::Newton;
      break;
    }
    case NonlinSolve::LINESEARCH: {
      SLIC_INFO_ROOT("using newton linesearch solver");
      nonlinear_options.min_iterations = 0;
      nonlinear_options.nonlin_solver  = NonlinearSolver::PetscNewtonBacktracking;
      // nonlinear_options.nonlin_solver = NonlinearSolver::NewtonLineSearch;
      break;
    }
    case NonlinSolve::CRITICALPOINT: {
      SLIC_INFO_ROOT("using newton critical point solver");
      nonlinear_options.min_iterations = 0;
      nonlinear_options.nonlin_solver  = NonlinearSolver::PetscNewtonCriticalPoint;
      break;
    }
    case NonlinSolve::TRUSTREGION: {
      SLIC_INFO_ROOT("using trust region solver");
      nonlinear_options.nonlin_solver = NonlinearSolver::TrustRegion;
      break;
    }
  }

  switch (prec) {
    case Prec::JACOBI: {
      SLIC_INFO_ROOT("using jacobi");
      linear_options.linear_solver  = LinearSolver::CG;
      linear_options.preconditioner = Preconditioner::HypreJacobi;
      break;
    }
    case Prec::STRUMPACK: {
      SLIC_INFO_ROOT("using strumpack");
      linear_options.linear_solver = LinearSolver::Strumpack;
      break;
    }
    case Prec::CHOLESKI: {
      SLIC_INFO_ROOT("using choleski");
      linear_options.linear_solver        = LinearSolver::CG;
      linear_options.preconditioner       = Preconditioner::Petsc;
      linear_options.petsc_preconditioner = PetscPCType::CHOLESKY;
      break;
    }
    case Prec::LU: {
      SLIC_INFO_ROOT("using lu");
      linear_options.linear_solver        = LinearSolver::GMRES;
      linear_options.preconditioner       = Preconditioner::Petsc;
      linear_options.petsc_preconditioner = PetscPCType::LU;
      break;
    }
    case Prec::MULTIGRID: {
      SLIC_INFO_ROOT("using multigrid");
      linear_options.linear_solver  = LinearSolver::CG;
      linear_options.preconditioner = Preconditioner::HypreAMG;
      break;
    }
    case Prec::PETSC_MULTIGRID: {
      SLIC_INFO_ROOT("using petsc multigrid");
      linear_options.linear_solver        = LinearSolver::CG;
      linear_options.preconditioner       = Preconditioner::Petsc;
      linear_options.petsc_preconditioner = PetscPCType::HMG;
      break;
    }
    default: {
      SLIC_ERROR_ROOT("error, invalid preconditioner specified");
    }
  }

  return std::make_pair(nonlinear_options, linear_options);
}

#include <ostream>
#include <fstream>

void functional_solid_test_euler()
{
  // initialize serac
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "eulerStore");

  static constexpr int ORDER{1};
  static constexpr int DIM{3};

  int Nx = 4;
  int Ny = 7;
  int Nz = 10 * 5;

  double Lx = 1.0;
  double Ly = 1.2;
  double Lz = 30.0;

  double density  = 1.0;
  double E        = 10.0;
  double v        = 0.33;
  double bulkMod  = E / (3. * (1. - 2. * v));
  double shearMod = E / (2. * (1. + v));
  double load     = 0.002;  // 0.004

  std::string    meshTag = "mesh";
  mfem::Mesh     mesh    = mfem::Mesh::MakeCartesian3D(Nx, Ny, Nz, mfem::Element::HEXAHEDRON, Lx, Ly, Lz);
  auto           pmesh   = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, mesh);
  mfem::ParMesh* meshPtr = &serac::StateManager::setMesh(std::move(pmesh), meshTag);

  // solid mechanics
  using seracSolidType = serac::SolidMechanics<ORDER, DIM, serac::Parameters<>>;

  auto [nonlinear_options, linear_options] = get_opts(3 * Nx * Ny * Nz, 1e-9);

  auto seracSolid = std::make_unique<seracSolidType>(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options,
      serac::GeometricNonlinearities::On, "serac_solid", meshTag, std::vector<std::string>{});

  serac::solid_mechanics::NeoHookean material{density, bulkMod, shearMod};
  seracSolid->setMaterial(serac::DependsOn<>{}, material);

  serac::Domain backSurface =
      serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(3));  // 4,5 with traction makes a twist
  serac::Domain topSurface = serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(6));

  int    num_time_steps = 2;
  double total_time     = 1.0;
  double dt             = total_time / num_time_steps;

  seracSolid->setTraction([&](auto, auto n, auto t) { return -load * t * n; }, topSurface);
  seracSolid->setTraction([&](auto, auto n, auto) { return 1e-5 * n; }, backSurface);

  // displacement on bottom surface
  seracSolid->setDisplacementBCs({1}, [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; });

  seracSolid->completeSetup();

  seracSolid->outputStateToDisk("paraview_euler");
  for (int i = 0; i < num_time_steps; ++i) {
    seracSolid->advanceTimestep(dt);
    seracSolid->outputStateToDisk("paraview_euler");
  }
}

void functional_solid_test_nonlinear_buckle(double loadMagnitude)
{
  // initialize serac
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "buckleStore");

  static constexpr int ORDER{1};
  static constexpr int DIM{3};

  // int Nx = 1000;
  int Nx = 500;
  int Ny = 6;
  int Nz = 5;

  double Lx = Nx * 0.1;
  double Ly = Ny * 0.03;
  double Lz = Nz * 0.06;

  double density  = 1.0;
  double E        = 1.0;
  double v        = 0.33;
  double bulkMod  = E / (3. * (1. - 2. * v));
  double shearMod = E / (2. * (1. + v));

  SERAC_MARK_FUNCTION;

  std::string    meshTag = "mesh";
  mfem::Mesh     mesh    = mfem::Mesh::MakeCartesian3D(Nx, Ny, Nz, mfem::Element::HEXAHEDRON, Lx, Ly, Lz);
  auto           pmesh   = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, mesh);
  mfem::ParMesh* meshPtr = &serac::StateManager::setMesh(std::move(pmesh), meshTag);

  // solid mechanics
  using seracSolidType = serac::SolidMechanics<ORDER, DIM, serac::Parameters<>>;

  auto [nonlinear_options, linear_options] = get_opts(3 * Nx * Ny * Nz, 1e-11);

  auto seracSolid = std::make_unique<seracSolidType>(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options,
      serac::GeometricNonlinearities::On, "serac_solid", meshTag, std::vector<std::string>{});

  serac::solid_mechanics::NeoHookean material{density, bulkMod, shearMod};
  seracSolid->setMaterial(serac::DependsOn<>{}, material);

  // fix displacement on side surface
  seracSolid->setDisplacementBCs({2, 3, 4, 5}, [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; });
  // seracSolid->setDisplacementBCs({3}, [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; });

  serac::Domain topSurface = serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(6));
  // seracSolid->setTraction([&](auto, auto n, auto) { return -loadMagnitude * n; }, topSurface);
  seracSolid->setPressure([&](auto, auto) { return loadMagnitude; }, topSurface);
  seracSolid->completeSetup();
  seracSolid->advanceTimestep(1.0);

  seracSolid->outputStateToDisk("paraview_buckle_easy");
}

TEST(SolidMechanics, nonlinear_solve_buckle_easy) { functional_solid_test_nonlinear_buckle(5e-10); }
// TEST(SolidMechanics, nonlinear_solve_buckle_medium) { functional_solid_test_nonlinear_buckle(4e-4); }
// TEST(SolidMechanics, nonlinear_solve_buckle_hard) { functional_solid_test_nonlinear_buckle(3e-2); }
// TEST(SolidMechanics, nonlinear_solve_euler) { functional_solid_test_euler(); }

int main(int argc, char* argv[])
{
  axom::CLI::App app{"Nonlinear problems"};
  // app.add_option("-p", mesh_path, "Path to mesh files")->check(axom::CLI::ExistingDirectory);
  app.add_option("--nonlinear-solver", nonlinSolve, "Nonlinear solver", true);
  app.add_option("--preconditioner", prec, "Preconditioner", true);
  app.set_help_flag("--help");
  app.allow_extras()->parse(argc, argv);

  ::testing::InitGoogleTest(&argc, argv);

  serac::initialize(argc, argv);

  SERAC_SET_METADATA("test", "solid_nonlinear_solve");
  SERAC_SET_METADATA("nonlinear solver", std::to_string(nonlinSolve));
  SERAC_SET_METADATA("preconditioner", std::to_string(prec));

  double x = 1.0 / 0.0;
  std::cout << "x = " << x << std::endl;

  int result = RUN_ALL_TESTS();
  serac::exitGracefully(result);

  return result;
}

// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/thermal_solid_functional.hpp"
#include "serac/physics/materials/thermal_functional_material.hpp"
#include "serac/physics/materials/solid_functional_material.hpp"
#include "serac/physics/materials/green_saint_venant_thermoelastic.hpp"

#include <fstream>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"

namespace serac {

template <int p>
void functional_test_static(double expected_norm)
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int dim{3};
  int serial_refinement   = 1;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "thermal_solid_functional_static_solve");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  // Define a boundary attribute set
  std::set<int> ess_bdr = {1};

  // define the thermal solver configurations
  auto thermal_options = Thermal::defaultQuasistaticOptions();

  // define the solid solver configurations
  // no default solver options for solid yet, so make some here
  const IterativeSolverOptions default_linear_options = {.rel_tol     = 1.0e-6,
                                                         .abs_tol     = 1.0e-10,
                                                         .print_level = 0,
                                                         .max_iter    = 500,
                                                         .lin_solver  = LinearSolver::GMRES,
                                                         .prec        = HypreBoomerAMGPrec{}};

  const NonlinearSolverOptions default_nonlinear_options = {
      .rel_tol = 1.0e-4, .abs_tol = 1.0e-8, .max_iter = 10, .print_level = 1};

  const typename solid_util::SolverOptions solid_options = {default_linear_options, default_nonlinear_options};

  // Construct a functional-based thermal-solid solver
  // BT 04/27/2022 This can't be instantiated yet.
  // The material model needs to be implemented before this
  // module can be used.
  ThermalMechanicsFunctional<p, dim> thermal_solid_solver(thermal_options, solid_options,
                                                          GeometricNonlinearities::On,
                                                          FinalMeshOption::Deformed, "thermal_solid_functional");
  double rho = 1.0;
  double E = 1.0;
  double nu = 0.25;
  double c = 1.0;
  double alpha = 1.0e-3;
  double theta_ref = 300.0;
  double k = 1.0;
  GreenSaintVenantThermoelasticMaterial material{rho, E, nu, c, alpha, theta_ref, k};
  thermal_solid_solver.setMaterial(material);

  // Define the function for the initial temperature and boundary condition
  auto one = [](const mfem::Vector&, double) -> double { return 1.0; };

  // Set the initial temperature and boundary condition
  thermal_solid_solver.setTemperatureBCs(ess_bdr, one);
  thermal_solid_solver.setTemperature(one);

  // Define the function for the disolacement boundary condition
  auto zeroVector = [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; };
  
  // Set the initial displcament and boundary condition
  thermal_solid_solver.setDisplacementBCs(ess_bdr, zeroVector);

  // Finalize the data structures
  thermal_solid_solver.completeSetup();

  // Perform the quasi-static solve
  double dt = 1.0;
  thermal_solid_solver.advanceTimestep(dt);

  // Output the sidre-based plot files
  // thermal_solid_solver.outputState();

  // Check the final temperature norm
  EXPECT_NEAR(expected_norm, norm(thermal_solid_solver.temperature()), 1.0e-6);
}

TEST(thermal_solid_functional, construct) { functional_test_static<1>(0.0); }

}  // namespace serac

//------------------------------------------------------------------------------
#include "axom/slic/core/SimpleLogger.hpp"

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}

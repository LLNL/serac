// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/thermal_solid.hpp"

#include <fstream>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/physics/coefficients/coefficient_extensions.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/serac_config.hpp"

namespace serac {

TEST(solid_solver, thermal_expansion)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_thermal_expansion");

  // Open the mesh
  std::string mesh_file = std::string(SERAC_REPO_DIR) + "/data/meshes/onehex.mesh";

  auto pmesh = mesh::refineAndDistribute(buildMeshFromFile(mesh_file), 1, 0);
  serac::StateManager::setMesh(std::move(pmesh));

  // define the solver configurations
  const IterativeSolverOptions default_linear_options = {.rel_tol     = 1.0e-10,
                                                         .abs_tol     = 1.0e-12,
                                                         .print_level = 0,
                                                         .max_iter    = 500,
                                                         .lin_solver  = LinearSolver::GMRES,
                                                         .prec        = HypreBoomerAMGPrec{}};

  const NonlinearSolverOptions default_nonlinear_options = {
      .rel_tol = 1.0e-4, .abs_tol = 1.0e-8, .max_iter = 500, .print_level = 1};

  const Solid::SolverOptions default_static = {default_linear_options, default_nonlinear_options};

  // initialize the solver object
  Solid solid_solver(1, default_static, GeometricNonlinearities::Off);

  solid_solver.setMaterialParameters(std::make_unique<mfem::ConstantCoefficient>(0.25),
                                     std::make_unique<mfem::ConstantCoefficient>(5.0), false);

  // set the boundary conditions to be fixed on the coordinate planes
  auto zero = std::make_shared<mfem::ConstantCoefficient>(0.0);

  solid_solver.setDisplacementBCs({1}, zero, 0);
  solid_solver.setDisplacementBCs({2}, zero, 1);
  solid_solver.setDisplacementBCs({3}, zero, 2);

  solid_solver.displacement() = 0.0;

  // Make a dummy temperature finite element state to drive the thermal expansion
  auto temp = serac::StateManager::newState(FiniteElementVector::Options{.name = "temp"});

  temp = 2.0;

  // Define the thermal expansion model
  auto ref_temp = std::make_unique<mfem::ConstantCoefficient>(1.0);
  auto cte      = std::make_unique<mfem::ConstantCoefficient>(0.1);

  solid_solver.setThermalExpansion(std::move(cte), std::move(ref_temp), temp);

  // Initialize the VisIt output
  solid_solver.initializeOutput(serac::OutputType::VisIt, "solid_thermal_expansion");

  // Construct the internal dynamic solver data structures
  solid_solver.completeSetup();

  // Ouput the initial state
  solid_solver.outputState();

  double t = 0.0;

  // Solve the quasi-static thermal expansion problem
  solid_solver.advanceTimestep(t);

  // Output the deformed state
  solid_solver.outputState();

  // Check the norm of the displacement error
  EXPECT_NEAR(0.11536897, norm(solid_solver.displacement()), 1.0e-4);

  MPI_Barrier(MPI_COMM_WORLD);
}

}  // namespace serac

//------------------------------------------------------------------------------
#include "axom/slic/core/SimpleLogger.hpp"

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;  // create & initialize test logger, finalized when
                                    // exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}

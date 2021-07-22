// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid.hpp"

#include <fstream>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/coefficients/coefficient_extensions.hpp"
#include "serac/infrastructure/input.hpp"
#include "serac/numerics/mesh_utils.hpp"
#include "serac/physics/utilities/state_manager.hpp"
#include "serac/serac_config.hpp"
#include "test_utilities.hpp"

namespace serac {

TEST(solid_solver, reuse_mesh)
{
  MPI_Barrier(MPI_COMM_WORLD);

  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore);

  // Open the mesh
  std::string mesh_file = std::string(SERAC_REPO_DIR) + "/data/meshes/beam-hex.mesh";
  auto        pmesh     = mesh::refineAndDistribute(buildMeshFromFile(mesh_file), 0, 0);
  const int   dim       = pmesh->Dimension();
  serac::StateManager::setMesh(std::move(pmesh));

  // define a boundary attribute set
  std::set<int> ess_bdr = {1};

  auto deform = std::make_shared<mfem::VectorFunctionCoefficient>(dim, [](const mfem::Vector& x, mfem::Vector& y) {
    y    = 0.0;
    y(1) = x(0) * 0.001;
  });

  // set the traction boundary
  std::set<int> trac_bdr = {2};

  // define the traction vector
  mfem::Vector traction(dim);
  traction           = 0.0;
  traction(1)        = 1.0e-5;
  auto traction_coef = std::make_shared<mfem::VectorConstantCoefficient>(traction);

  // Use the same configuration as the solid solver
  const IterativeSolverOptions default_linear_options = {.rel_tol     = 1.0e-4,
                                                         .abs_tol     = 1.0e-8,
                                                         .print_level = 0,
                                                         .max_iter    = 500,
                                                         .lin_solver  = LinearSolver::GMRES,
                                                         .prec        = HypreBoomerAMGPrec{}};

  const NonlinearSolverOptions default_nonlinear_options = {
      .rel_tol = 1.0e-4, .abs_tol = 1.0e-8, .max_iter = 500, .print_level = 1};

  const Solid::SolverOptions default_static = {default_linear_options, default_nonlinear_options};

  mfem::Vector u_1_true_vec;

  // Keep the solver_1 and solver_2 objects in a different scope for testing
  {
    // initialize the dynamic solver object
    Solid solid_solver_1(1, default_static, GeometricNonlinearities::On, FinalMeshOption::Deformed, "first_solid");
    solid_solver_1.setDisplacementBCs(ess_bdr, deform);
    solid_solver_1.setTractionBCs(trac_bdr, traction_coef, false);
    solid_solver_1.setMaterialParameters(std::make_unique<mfem::ConstantCoefficient>(0.25),
                                         std::make_unique<mfem::ConstantCoefficient>(5.0));
    solid_solver_1.setDisplacement(*deform);

    // Initialize the VisIt output
    solid_solver_1.initializeOutput(serac::OutputType::VisIt, "two_mesh_solid_1");

    // Construct the internal dynamic solver data structures
    solid_solver_1.completeSetup();

    Solid solid_solver_2(1, default_static, GeometricNonlinearities::On, FinalMeshOption::Deformed, "second_solid");
    solid_solver_2.setDisplacementBCs(ess_bdr, deform);
    solid_solver_2.setTractionBCs(trac_bdr, traction_coef, false);
    solid_solver_2.setMaterialParameters(std::make_unique<mfem::ConstantCoefficient>(0.25),
                                         std::make_unique<mfem::ConstantCoefficient>(5.0));
    solid_solver_2.setDisplacement(*deform);

    // Initialize the VisIt output
    solid_solver_2.initializeOutput(serac::OutputType::VisIt, "two_mesh_solid_2");

    // Construct the internal dynamic solver data structures
    solid_solver_2.completeSetup();

    double dt = 1.0;
    solid_solver_1.advanceTimestep(dt);
    solid_solver_2.advanceTimestep(dt);

    // Output the final state
    solid_solver_1.outputState();

    u_1_true_vec = solid_solver_1.displacement().trueVec();

    EXPECT_NEAR(
        0.0, (mfem::Vector(solid_solver_1.displacement().trueVec() - solid_solver_2.displacement().trueVec())).Norml2(),
        0.001);
  }

  Solid solid_solver_3(1, default_static, GeometricNonlinearities::On, FinalMeshOption::Deformed);
  solid_solver_3.setDisplacementBCs(ess_bdr, deform);
  solid_solver_3.setTractionBCs(trac_bdr, traction_coef, false);
  solid_solver_3.setMaterialParameters(std::make_unique<mfem::ConstantCoefficient>(0.25),
                                       std::make_unique<mfem::ConstantCoefficient>(5.0));
  solid_solver_3.setDisplacement(*deform);

  // Initialize the VisIt output
  solid_solver_3.initializeOutput(serac::OutputType::VisIt, "two_mesh_solid_1");

  // Construct the internal dynamic solver data structures
  solid_solver_3.completeSetup();

  double dt = 1.0;
  solid_solver_3.advanceTimestep(dt);

  EXPECT_NEAR(0.0, (mfem::Vector(u_1_true_vec - solid_solver_3.displacement().trueVec())).Norml2(), 0.001);

  solid_solver_3.resetToReferenceConfiguration();
  EXPECT_NEAR(0.0, norm(solid_solver_3.displacement()), 1.0e-8);
  EXPECT_NEAR(0.0, norm(solid_solver_3.velocity()), 1.0e-8);

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

  axom::slic::SimpleLogger logger;  // create & initialize test logger, finalized when exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}

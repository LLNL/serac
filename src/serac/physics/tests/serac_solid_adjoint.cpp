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
#include "serac/physics/state/state_manager.hpp"
#include "serac/serac_config.hpp"
#include "test_utilities.hpp"

namespace serac {

TEST(solid_solver, adjoint)
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

  // define the solver configurations
  const IterativeSolverOptions default_linear_options = {.rel_tol     = 1.0e-6,
                                                         .abs_tol     = 1.0e-10,
                                                         .print_level = 0,
                                                         .max_iter    = 500,
                                                         .lin_solver  = LinearSolver::GMRES,
                                                         .prec        = HypreBoomerAMGPrec{}};

  const NonlinearSolverOptions default_nonlinear_options = {
      .rel_tol = 1.0e-4, .abs_tol = 1.0e-8, .max_iter = 500, .print_level = 1};

  const Solid::SolverOptions default_static = {default_linear_options, default_nonlinear_options};

  // initialize the solver object
  Solid solid_solver(1, default_static, GeometricNonlinearities::Off, FinalMeshOption::Deformed, "first_solve");
  solid_solver.setDisplacementBCs(ess_bdr, deform);
  solid_solver.setTractionBCs(trac_bdr, traction_coef, false);
  solid_solver.setMaterialParameters(std::make_unique<mfem::ConstantCoefficient>(0.25),
                                     std::make_unique<mfem::ConstantCoefficient>(5.0), false);
  solid_solver.setDisplacement(*deform);

  // Construct the internal solver data structures
  solid_solver.completeSetup();

  double dt = 1.0;
  solid_solver.advanceTimestep(dt);

  // Save the first true vector
  mfem::Vector true_vec_1 = solid_solver.displacement().trueVec();

  // Make a dummy adjoint load for testing

  FiniteElementDual assembled_adjoint_load(StateManager::mesh(), solid_solver.displacement().space(), "adjoint_load");

  mfem::ParLinearForm adjoint_load(&assembled_adjoint_load.space());

  mfem::Vector load(dim);
  load = 0.1;
  mfem::VectorConstantCoefficient loadvec(load);
  adjoint_load.AddDomainIntegrator(new mfem::VectorDomainLFIntegrator(loadvec));

  adjoint_load.Assemble();
  assembled_adjoint_load.trueVec() = *adjoint_load.ParallelAssemble();
  assembled_adjoint_load.distributeSharedDofs();

  auto&  adjoint_state_1 = solid_solver.solveAdjoint(assembled_adjoint_load);
  double adjoint_norm_1  = norm(adjoint_state_1);

  SLIC_INFO_ROOT(fmt::format("Adjoint norm (homogeneous BCs): {}", adjoint_norm_1));

  // Create an L2 space for the shear modulus discretization
  mfem::L2_FECollection       l2_fe_coll(0, dim);
  mfem::ParFiniteElementSpace l2_fe_space(&serac::StateManager::mesh(), &l2_fe_coll);

  // Compute, assemble, and check the sensitivities
  auto& shear_sensitivity = solid_solver.shearModulusSensitivity(&l2_fe_space);

  double shear_norm = mfem::ParNormlp(shear_sensitivity.trueVec(), 2, MPI_COMM_WORLD);

  SLIC_INFO_ROOT(fmt::format("Shear sensitivity vector norm: {}", shear_norm));

  EXPECT_NEAR(shear_norm, 0.000211078850122, 1.0e-8);

  auto& bulk_sensitivity = solid_solver.bulkModulusSensitivity(&l2_fe_space);

  double bulk_norm = mfem::ParNormlp(bulk_sensitivity.trueVec(), 2, MPI_COMM_WORLD);

  SLIC_INFO_ROOT(fmt::format("Bulk sensitivity vector norm: {}", bulk_norm));

  EXPECT_NEAR(bulk_norm, 6.026196496111377e-06, 3.0e-9);

  // Do a forward solve again to make sure the adjoint solve didn't break the solver
  solid_solver.setDisplacement(*deform);
  solid_solver.advanceTimestep(dt);

  // Check that the two forward solves are equal
  EXPECT_NEAR(0.0, (mfem::Vector(true_vec_1 - solid_solver.displacement().trueVec())).Norml2(), 0.00001);

  // Check that the adjoint solve is a known value
  EXPECT_NEAR(adjoint_norm_1, 738.4103079, 0.05);

  // Do another adjoint solve with a non-homogeneous BC
  // Also test the state manager option for finite element duals
  auto adjoint_essential =
      StateManager::newDual(FiniteElementVector::Options{.vector_dim = dim, .name = "adjoint_essential_load"});

  // Set the essential boundary to a non-zero value
  adjoint_essential = 0.5;

  auto&  adjoint_state_2 = solid_solver.solveAdjoint(assembled_adjoint_load, &adjoint_essential);
  double adjoint_norm_2  = norm(adjoint_state_2);

  SLIC_INFO_ROOT(fmt::format("Adjoint norm (non-homogeneous BCs): {}", adjoint_norm_2));

  // Check that the adjoint solve is a known value
  EXPECT_NEAR(adjoint_norm_2, 739.98432, 0.05);

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

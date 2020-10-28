// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include <fstream>

#include "mfem.hpp"
#include "numerics/mesh_utils.hpp"
#include "physics/elasticity.hpp"
#include "serac_config.hpp"

namespace serac {

TEST(elastic_solver, static_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Open the mesh
  std::string mesh_file = std::string(SERAC_REPO_DIR) + "/data/meshes/beam-quad.mesh";

  auto pmesh = buildMeshFromFile(mesh_file, 1, 0);

  IterativeSolverParameters default_quasistatic = {.rel_tol     = 1.0e-4,
                                                   .abs_tol     = 1.0e-10,
                                                   .print_level = 0,
                                                   .max_iter    = 500,
                                                   .lin_solver  = LinearSolver::MINRES,
                                                   .prec        = HypreSmootherPrec{mfem::HypreSmoother::l1Jacobi}};

  Elasticity elas_solver(1, pmesh, default_quasistatic);

  std::set<int> disp_bdr = {1};

  // define the displacement vector
  mfem::Vector disp(pmesh->Dimension());
  disp = 0.0;

  auto disp_coef = std::make_shared<mfem::VectorConstantCoefficient>(disp);
  elas_solver.setDisplacementBCs(disp_bdr, disp_coef);

  std::set<int> trac_bdr = {2};

  // define the traction vector
  mfem::Vector traction(pmesh->Dimension());
  traction           = 0.0;
  traction(1)        = 1.0e-4;
  auto traction_coef = std::make_shared<mfem::VectorConstantCoefficient>(traction);
  elas_solver.setTractionBCs(trac_bdr, traction_coef);

  // set the material properties
  mfem::ConstantCoefficient mu_coef(0.25);
  mfem::ConstantCoefficient K_coef(5.0);

  elas_solver.setLameParameters(K_coef, mu_coef);

  // allocate the data structures
  elas_solver.completeSetup();

  double dt = 1.0;
  elas_solver.advanceTimestep(dt);

  mfem::Vector zero(pmesh->Dimension());
  zero = 0.0;
  mfem::VectorConstantCoefficient zerovec(zero);

  double x_norm = elas_solver.getState()[0].get().gridFunc().ComputeLpError(2.0, zerovec);

  EXPECT_NEAR(0.128065, x_norm, 0.00001);

  MPI_Barrier(MPI_COMM_WORLD);
}

}  // namespace serac

//------------------------------------------------------------------------------
#include "axom/slic/core/UnitTestLogger.hpp"
using axom::slic::UnitTestLogger;

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  UnitTestLogger logger;  // create & initialize test logger, finalized when exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}

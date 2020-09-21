// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include <fstream>

#include "coefficients/loading_functions.hpp"
#include "mfem.hpp"
#include "numerics/mesh_utils.hpp"
#include "physics/nonlinear_solid_solver.hpp"
#include "serac_config.hpp"

namespace serac {

TEST(nonlinear_solid_solver, qs_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Open the mesh
  std::string mesh_file = std::string(SERAC_REPO_DIR) + "/data/beam-hex.mesh";

  auto pmesh = buildMeshFromFile(mesh_file, 1, 0);

  int dim = pmesh->Dimension();

  // Define the solver object
  NonlinearSolidSolver solid_solver(1, pmesh);

  std::set<int> ess_bdr = {1};

  // define the displacement vector
  mfem::Vector disp(dim);
  disp = 0.0;

  auto disp_coef = std::make_shared<mfem::VectorConstantCoefficient>(disp);

  std::set<int> trac_bdr = {2};

  // define the traction vector
  mfem::Vector traction(dim);
  traction           = 0.0;
  traction(1)        = 1.0e-3;
  auto traction_coef = std::make_shared<mfem::VectorConstantCoefficient>(traction);

  // Pass the BC information to the solver object
  solid_solver.setDisplacementBCs(ess_bdr, disp_coef);
  solid_solver.setTractionBCs(trac_bdr, traction_coef);

  // Set the material parameters
  solid_solver.setHyperelasticMaterialParameters(0.25, 10.0);

  // Set the linear solver params
  serac::LinearSolverParameters params;
  params.rel_tol     = 1.0e-6;
  params.abs_tol     = 1.0e-8;
  params.print_level = 0;
  params.max_iter    = 5000;
  params.prec        = serac::Preconditioner::Jacobi;
  params.lin_solver  = serac::LinearSolver::MINRES;

  serac::NonlinearSolverParameters nl_params;
  nl_params.rel_tol     = 1.0e-3;
  nl_params.abs_tol     = 1.0e-6;
  nl_params.print_level = 1;
  nl_params.max_iter    = 5000;

  solid_solver.setSolverParameters(params, nl_params);

  // Set the time step method
  solid_solver.setTimestepper(serac::TimestepMethod::QuasiStatic);

  // Initialize the output
  solid_solver.initializeOutput(serac::OutputType::VisIt, "static_solid");

  // Complete the solver setup
  solid_solver.completeSetup();

  double dt = 1.0;
  solid_solver.advanceTimestep(dt);

  solid_solver.outputState();

  mfem::Vector zero(dim);
  zero = 0.0;
  mfem::VectorConstantCoefficient zerovec(zero);

  double x_norm = solid_solver.displacement()->gridFunc().ComputeLpError(2.0, zerovec);

  EXPECT_NEAR(2.2309025, x_norm, 0.001);

  MPI_Barrier(MPI_COMM_WORLD);
}

/*
TODO this test is disabled as it was failing CI due to a memory leak in MFEM.
Once that leak is fixed, it should be re-enabled

TEST(nonlinear_solid_solver, qs_direct_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Open the mesh
  std::string mesh_file = std::string(SERAC_REPO_DIR) + "/data/beam-hex.mesh";

  auto pmesh = buildMeshFromFile(mesh_file, 1, 0);

  int dim = pmesh->Dimension();

  // Define the solver object
  NonlinearSolidSolver solid_solver(1, pmesh);

  std::set<int> ess_bdr = {1};

  // define the displacement vector
  mfem::Vector disp(dim);
  disp = 0.0;

  auto disp_coef = std::make_shared<mfem::VectorConstantCoefficient>(disp);

  std::set<int> trac_bdr = {2};

  // define the traction vector
  mfem::Vector traction(dim);
  traction           = 0.0;
  traction(1)        = 1.0e-3;
  auto traction_coef = std::make_shared<mfem::VectorConstantCoefficient>(traction);

  // Pass the BC information to the solver object
  solid_solver.setDisplacementBCs(ess_bdr, disp_coef);
  solid_solver.setTractionBCs(trac_bdr, traction_coef);

  // Set the material parameters
  solid_solver.setHyperelasticMaterialParameters(0.25, 10.0);

  // Set the linear solver params
  serac::LinearSolverParameters params;
  params.lin_solver = serac::LinearSolver::SuperLU;

  serac::NonlinearSolverParameters nl_params;
  nl_params.rel_tol     = 1.0e-3;
  nl_params.abs_tol     = 1.0e-6;
  nl_params.print_level = 1;
  nl_params.max_iter    = 5000;

  solid_solver.setSolverParameters(params, nl_params);

  // Set the time step method
  solid_solver.setTimestepper(serac::TimestepMethod::QuasiStatic);

  // Initialize the output
  solid_solver.initializeOutput(serac::OutputType::VisIt, "static_solid");

  // Complete the solver setup
  solid_solver.completeSetup();

  double dt = 1.0;
  solid_solver.advanceTimestep(dt);

  solid_solver.outputState();

  mfem::Vector zero(dim);
  zero = 0.0;
  mfem::VectorConstantCoefficient zerovec(zero);

  double x_norm = solid_solver.displacement()->gridFunc().ComputeLpError(2.0, zerovec);

  EXPECT_NEAR(2.2309025, x_norm, 0.001);

  MPI_Barrier(MPI_COMM_WORLD);
}
*/

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

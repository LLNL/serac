// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include <fstream>

#include "mfem.hpp"
#include "serac_config.hpp"
#include "solvers/elasticity_solver.hpp"

TEST(elastic_solver, static_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // mesh
  std::string base_mesh_file = std::string(SERAC_SRC_DIR) + "/data/beam-quad.mesh";
  const char* mesh_file      = base_mesh_file.c_str();

  // Open the mesh
  std::ifstream imesh(mesh_file);
  auto          mesh = std::make_unique<mfem::Mesh>(imesh, 1, 1, true);
  imesh.close();

  // declare pointer to parallel mesh object
  mesh->UniformRefinement();

  auto pmesh = std::make_shared<mfem::ParMesh>(MPI_COMM_WORLD, *mesh);

  ElasticitySolver elas_solver(1, pmesh);

  // define a boundary attribute array and initialize to 0
  std::vector<int> disp_bdr(pmesh->bdr_attributes.Max(), 0);

  // boundary attribute 1 (index 0) is fixed (Dirichlet)
  disp_bdr[0] = 1;

  // define the displacement vector
  mfem::Vector disp(pmesh->Dimension());
  disp           = 0.0;
  auto disp_coef = std::make_shared<mfem::VectorConstantCoefficient>(disp);
  elas_solver.SetDisplacementBCs(disp_bdr, disp_coef);

  std::vector<int> trac_bdr(pmesh->bdr_attributes.Max(), 0);
  trac_bdr[1] = 1;

  // define the traction vector
  mfem::Vector traction(pmesh->Dimension());
  traction           = 0.0;
  traction(1)        = 1.0e-4;
  auto traction_coef = std::make_shared<mfem::VectorConstantCoefficient>(traction);
  elas_solver.SetTractionBCs(trac_bdr, traction_coef);

  // set the material properties
  mfem::ConstantCoefficient mu_coef(0.25);
  mfem::ConstantCoefficient K_coef(5.0);

  elas_solver.SetLameParameters(K_coef, mu_coef);

  // Define the linear solver params
  LinearSolverParameters params;
  params.rel_tol     = 1.0e-4;
  params.abs_tol     = 1.0e-10;
  params.print_level = 0;
  params.max_iter    = 500;
  params.prec        = Preconditioner::Jacobi;
  params.lin_solver  = LinearSolver::MINRES;

  elas_solver.SetLinearSolverParameters(params);
  elas_solver.SetTimestepper(TimestepMethod::QuasiStatic);

  // allocate the data structures
  elas_solver.CompleteSetup();

  double dt = 1.0;
  elas_solver.AdvanceTimestep(dt);

  auto state = elas_solver.GetState();

  mfem::Vector zero(pmesh->Dimension());
  zero = 0.0;
  mfem::VectorConstantCoefficient zerovec(zero);

  double x_norm = state[0].gf->ComputeLpError(2.0, zerovec);

  EXPECT_NEAR(0.128065, x_norm, 0.00001);

  MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}

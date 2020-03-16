// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include "mfem.hpp"
#include "solvers/elasticity_solver.hpp"
#include <fstream>

const char* mesh_file = "NO_MESH_GIVEN";

inline bool file_exists(const char* path)
{
  struct stat buffer;
  return (stat(path, &buffer) == 0);
}

TEST(elastic_solver, static_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Open the mesh
  ASSERT_TRUE(file_exists(mesh_file));
  std::ifstream imesh(mesh_file);
  mfem::Mesh* mesh = new mfem::Mesh(imesh, 1, 1, true);
  imesh.close();

  // declare pointer to parallel mesh object
  mfem::ParMesh *pmesh = NULL;
  mesh->UniformRefinement();

  pmesh = new mfem::ParMesh(MPI_COMM_WORLD, *mesh);
  delete mesh;

  ElasticitySolver elas_solver(1, pmesh);

  // define a boundary attribute array and initialize to 0
  mfem::Array<int> disp_bdr;
  disp_bdr.SetSize(pmesh->bdr_attributes.Max());
  disp_bdr = 0;

  // boundary attribute 1 (index 0) is fixed (Dirichlet)
  disp_bdr[0] = 1;

  // define the displacement vector
  mfem::Vector disp(pmesh->Dimension());
  disp = 0.0;
  mfem::VectorConstantCoefficient disp_coef(disp);
  elas_solver.SetDisplacementBCs(disp_bdr, &disp_coef);

  mfem::Array<int> trac_bdr;
  trac_bdr.SetSize(pmesh->bdr_attributes.Max());
  trac_bdr = 0;
  trac_bdr[1] = 1;

  // define the traction vector
  mfem::Vector traction(pmesh->Dimension());
  traction = 0.0;
  traction(1) = 1.0e-4;
  mfem::VectorConstantCoefficient traction_coef(traction);
  elas_solver.SetTractionBCs(trac_bdr, &traction_coef);

  // set the material properties
  mfem::ConstantCoefficient mu_coef(0.25);
  mfem::ConstantCoefficient K_coef(5.0);

  elas_solver.SetLameParameters(K_coef, mu_coef);

  // Define the linear solver params
  LinearSolverParameters params;
  params.rel_tol = 1.0e-4;
  params.abs_tol = 1.0e-10;
  params.print_level = 0;
  params.max_iter = 500;
  params.prec = Preconditioner::Jacobi;
  params.lin_solver = LinearSolver::MINRES;

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

  double x_norm = state["displacement"].gf->ComputeLpError(2.0, zerovec);

  EXPECT_NEAR(0.128065, x_norm, 0.00001);

  delete pmesh;

  MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);
  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  // Parse command line options
  mfem::OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "Mesh file to use.", true);
  args.Parse();
  if (!args.Good()) {
    if (myid == 0) {
      args.PrintUsage(std::cout);
    }
    MPI_Finalize();
    return 1;
  }
  if (myid == 0) {
    args.PrintOptions(std::cout);
  }

  result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}

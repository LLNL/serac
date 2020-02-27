// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include "mfem.hpp"
#include "coefficients/loading_functions.hpp"
#include "solvers/nonlinear_solid_solver.hpp"
#include <fstream>

const char* mesh_file = "NO_MESH_GIVEN";

inline bool file_exists(const char* path)
{
  struct stat buffer;
  return (stat(path, &buffer) == 0);
}

TEST(nonlinear_solid_solver, qs_solve)
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

  int dim = pmesh->Dimension();

  // Define the solver object
  NonlinearSolidSolver solid_solver(1, pmesh);

  // define a boundary attribute array and initialize to 0
  mfem::Array<int> ess_bdr;
  ess_bdr.SetSize(pmesh->bdr_attributes.Max());
  ess_bdr = 0;

  // boundary attribute 1 (index 0) is fixed (Dirichlet)
  ess_bdr[0] = 1;

  // define the displacement vector
  mfem::Vector disp(dim);
  disp = 0.0;
  mfem::VectorConstantCoefficient disp_coef(disp);


  mfem::Array<int> trac_bdr;
  trac_bdr.SetSize(pmesh->bdr_attributes.Max());
  trac_bdr = 0;
  trac_bdr[1] = 1;

  // define the traction vector
  mfem::Vector traction(dim);
  traction = 0.0;
  traction(1) = 1.0e-3;
  mfem::VectorConstantCoefficient traction_coef(traction);

  // Pass the BC information to the solver object
  solid_solver.SetDisplacementBCs(ess_bdr, &disp_coef);
  solid_solver.SetTractionBCs(trac_bdr, &traction_coef);

  // Set the material parameters
  solid_solver.SetHyperelasticMaterialParameters(0.25, 10.0);

  // Set the linear solver params
  LinearSolverParameters params;
  params.rel_tol = 1.0e-3;
  params.abs_tol = 1.0e-6;
  params.print_level = 0;
  params.max_iter = 5000;
  params.prec = Preconditioner::Jacobi;
  params.lin_solver = LinearSolver::MINRES;
  solid_solver.SetLinearSolverParameters(params);
  
  // Set the time step method
  solid_solver.SetTimestepper(TimestepMethod::QuasiStatic);

  // Complete the solver setup
  solid_solver.CompleteSetup();

  double dt = 1.0;
  solid_solver.AdvanceTimestep(dt);

  auto state = solid_solver.GetState(); 

  mfem::Vector zero(dim);
  zero = 0.0;
  mfem::VectorConstantCoefficient zerovec(zero);

  double x_norm = state[0].gf->ComputeLpError(2.0, zerovec);

  EXPECT_NEAR(2.2322, x_norm, 0.001);

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

// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include <fstream>

#include "coefficients/stdfunction_coefficient.hpp"
#include "mfem.hpp"
#include "solvers/nonlinear_solid_solver.hpp"

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
  auto          mesh = std::make_unique<mfem::Mesh>(imesh, 1, 1, true);
  imesh.close();

  // declare pointer to parallel mesh object
  for (int i=0; i<3; ++i) {
    mesh->UniformRefinement();
  }

  auto pmesh = std::make_shared<mfem::ParMesh>(MPI_COMM_WORLD, *mesh);

  int dim = pmesh->Dimension();

  // Define the solver object
  NonlinearSolidSolver solid_solver(1, pmesh);

  // define a boundary attribute array and initialize to 0
  std::vector<int> ess_bdr(pmesh->bdr_attributes.Max(), 0);

  // boundary attribute 1 (index 0) is fixed (Dirichlet) in the x direction
  ess_bdr[0] = 1;

  // define the displacement vector
  auto disp_coef = std::make_shared<StdFunctionVectorCoefficient>(dim, [](mfem::Vector &x, mfem::Vector &X) {
    X = x;
    X[0] = x[0] * 5.0;
  });

  // Pass the BC information to the solver object setting only the z direction
  solid_solver.SetDisplacementBCs(ess_bdr, disp_coef, 0);

  // Create an indicator function to set all vertices that are x=0
  StdFunctionVectorCoefficient zero_bc(dim, [](mfem::Vector &x, mfem::Vector &X) {
    X = 0.;
    for (int i = 0; i < X.Size(); i++)
      if (abs(x[i]) < 1.e-13) {
        X[i] = 1.;
      }
  });

  mfem::Array<int> ess_corner_bc_list;
  MakeEssList(*pmesh, zero_bc, ess_corner_bc_list);

  solid_solver.SetTrueDofs(ess_corner_bc_list, disp_coef);

  // Set the material parameters
  solid_solver.SetHyperelasticMaterialParameters(0.25, 10.0);

  // Set the linear solver params
  LinearSolverParameters params;
  params.rel_tol     = 1.0e-6;
  params.abs_tol     = 1.0e-8;
  params.print_level = 0;
  params.max_iter    = 5000;
  params.prec        = Preconditioner::Jacobi;
  params.lin_solver  = LinearSolver::MINRES;

  NonlinearSolverParameters nl_params;
  nl_params.rel_tol     = 1.0e-3;
  nl_params.abs_tol     = 1.0e-6;
  nl_params.print_level = 1;
  nl_params.max_iter    = 5000;

  solid_solver.SetSolverParameters(params, nl_params);

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

  double x_norm = state[1].gf->ComputeLpError(2.0, zerovec);

  EXPECT_NEAR(2.2309025, x_norm, 0.001);

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
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.", true);
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

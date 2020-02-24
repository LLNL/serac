// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include "mfem.hpp"
#include "coefficients/loading_functions.hpp"
#include "solvers/quasistatic_solver.hpp"
#include <fstream>

const char* mesh_file = "NO_MESH_GIVEN";

inline bool file_exists(const char* path)
{
  struct stat buffer;
  return (stat(path, &buffer) == 0);
}

TEST(quasistatic_solver, qs_solve)
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

  // Define the finite element spaces for displacement field
  mfem::H1_FECollection fe_coll(1, dim);
  mfem::ParFiniteElementSpace fe_space(pmesh, &fe_coll, dim, mfem::Ordering::byVDIM);

  // Define a grid function for the global reference configuration, the beginning
  // step configuration, the global deformation, the current configuration/solution
  // guess, and the incremental nodal displacements
  mfem::ParGridFunction x_inc(&fe_space);

  mfem::VectorFunctionCoefficient defo_coef(dim, InitialDeformation);
  x_inc.ProjectCoefficient(defo_coef);
  x_inc.SetTrueVector();

  // define a boundary attribute array and initialize to 0
  mfem::Array<int> ess_bdr;
  ess_bdr.SetSize(fe_space.GetMesh()->bdr_attributes.Max());
  ess_bdr = 0;

  // boundary attribute 1 (index 0) is fixed (Dirichlet)
  ess_bdr[0] = 1;

  mfem::Array<int> trac_bdr;
  trac_bdr.SetSize(fe_space.GetMesh()->bdr_attributes.Max());

  trac_bdr = 0;
  trac_bdr[1] = 1;

  // define the traction vector
  mfem::Vector traction(dim);
  traction = 0.0;
  traction(1) = 1.0e-3;

  mfem::VectorConstantCoefficient traction_coef(traction);

  // construct the nonlinear mechanics operator
  QuasistaticSolver oper(fe_space, ess_bdr, trac_bdr,
                         0.25, 10.0, traction_coef,
                         1.0e-3, 1.0e-6,
                         5000, false, false);

  // declare incremental nodal displacement solution vector
  mfem::Vector x_sol(fe_space.TrueVSize());
  x_inc.GetTrueDofs(x_sol);

  // Solve the Newton system
  bool converged = oper.Solve(x_sol);

  // distribute the solution vector to x_cur
  x_inc.Distribute(x_sol);

  mfem::Vector zero(dim);
  zero = 0.0;
  mfem::VectorConstantCoefficient zerovec(zero);

  double x_norm = x_inc.ComputeLpError(2.0, zerovec);

  EXPECT_NEAR(2.2322, x_norm, 0.001);
  EXPECT_TRUE(converged);

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

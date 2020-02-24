// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include "mfem.hpp"
#include "solvers/dynamic_solver.hpp"
#include <fstream>

void InitialDeformation(const mfem::Vector &x, mfem::Vector &y);

void InitialVelocity(const mfem::Vector &x, mfem::Vector &v);

const char* mesh_file = "NO_MESH_GIVEN";

inline bool file_exists(const char* path)
{
  struct stat buffer;
  return (stat(path, &buffer) == 0);
}


TEST(dynamic_solver, dyn_solve)
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

  mfem::ODESolver *ode_solver = new mfem::SDIRK33Solver;

  // Define the finite element spaces for displacement field
  mfem::H1_FECollection fe_coll(1, dim);
  mfem::ParFiniteElementSpace fe_space(pmesh, &fe_coll, dim, mfem::Ordering::byVDIM);

  int true_size = fe_space.TrueVSize();
  mfem::Array<int> true_offset(3);
  true_offset[0] = 0;
  true_offset[1] = true_size;
  true_offset[2] = 2*true_size;

  mfem::BlockVector vx(true_offset);
  mfem::ParGridFunction v_gf, x_gf;
  v_gf.MakeTRef(&fe_space, vx, true_offset[0]);
  x_gf.MakeTRef(&fe_space, vx, true_offset[1]);

  mfem::VectorFunctionCoefficient velo_coef(dim, InitialVelocity);
  v_gf.ProjectCoefficient(velo_coef);
  v_gf.SetTrueVector();

  mfem::VectorFunctionCoefficient deform(dim, InitialDeformation);
  x_gf.ProjectCoefficient(deform);
  x_gf.SetTrueVector();

  v_gf.SetFromTrueVector();
  x_gf.SetFromTrueVector();


  // define a boundary attribute array and initialize to 0
  mfem::Array<int> ess_bdr;
  ess_bdr.SetSize(fe_space.GetMesh()->bdr_attributes.Max());
  ess_bdr = 0;

  // boundary attribute 1 (index 0) is fixed (Dirichlet)
  ess_bdr[0] = 1;

  mfem::ConstantCoefficient visc(0.0);

  // construct the nonlinear mechanics operator
  DynamicSolver oper(fe_space, ess_bdr,
                     0.25, 5.0, visc,
                     1.0e-4, 1.0e-8,
                     500, true, false);

  double t = 0.0;
  double t_final = 6.0;
  double dt = 3.0;

  oper.SetTime(t);
  ode_solver->Init(oper);

  // Perform time-integration
  // (looping over the time iterations, ti, with a time-step dt).
  bool last_step = false;
  for (int ti = 1; !last_step; ti++) {
    double dt_real = std::min(dt, t_final - t);

    ode_solver->Step(vx, t, dt_real);

    last_step = (t >= t_final - 1e-8*dt);
  }

  mfem::Vector zero(dim);
  zero = 0.0;
  mfem::VectorConstantCoefficient zerovec(zero);

  double x_norm = x_gf.ComputeLpError(2.0, zerovec);
  double v_norm = v_gf.ComputeLpError(2.0, zerovec);

  EXPECT_NEAR(13.2665, x_norm, 0.0001);
  EXPECT_NEAR(0.25368, v_norm, 0.0001);

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

void InitialDeformation(const mfem::Vector &x, mfem::Vector &y)
{
  // set the initial configuration to be the same as the reference, stress
  // free, configuration
  y = x;
}

void InitialVelocity(const mfem::Vector &x, mfem::Vector &v)
{
  const int dim = x.Size();
  const double s = 0.1/64.;

  v = 0.0;
  v(dim-1) = s*x(0)*x(0)*(8.0-x(0));
  v(0) = -s*x(0)*x(0);
}

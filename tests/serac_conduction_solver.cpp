// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include "mfem.hpp"
#include "solvers/conduction_solver.hpp"
#include <fstream>
#include <sys/stat.h>

double InitialTemperature(const mfem::Vector &x);

const char* mesh_file = "NO_MESH_GIVEN";

inline bool file_exists(const char* path) {
  struct stat buffer;   
  return (stat(path, &buffer) == 0); 
}


TEST(dynamic_solver, dyn_solve)
{
   MPI_Barrier(MPI_COMM_WORLD);

   // Open the mesh
   ASSERT_TRUE(file_exists(mesh_file));
   std::fstream imesh(mesh_file);
   mfem::Mesh* mesh = new mfem::Mesh(imesh, 1, 1, true);
   imesh.close();

   // declare pointer to parallel mesh object
   mfem::ParMesh *pmesh = NULL;
   mesh->UniformRefinement();

   pmesh = new mfem::ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   pmesh->UniformRefinement();

   int dim = pmesh->Dimension();

   mfem::ODESolver *ode_solver = new mfem::BackwardEulerSolver;

   // Define the finite element spaces for temperature field
   mfem::H1_FECollection fe_coll(2, dim);
   mfem::ParFiniteElementSpace fe_space(pmesh, &fe_coll);

   mfem::ParGridFunction u_gf(&fe_space);

   mfem::FunctionCoefficient u_0(InitialTemperature);
   u_gf.ProjectCoefficient(u_0);
   mfem::Vector u;
   u_gf.GetTrueDofs(u);
   u_gf.SetFromTrueDofs(u);

   int myid;
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // Initialize the conduction operator and the VisIt visualization.
   mfem::ConstantCoefficient kappa(0.5);
   ConductionSolver oper(fe_space, kappa);

   ode_solver->Init(oper);
   double t = 0.0;
   double t_final = 5.0;
   double dt = 1.0;

   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final - dt/2)
      {
         last_step = true;
      }
      ode_solver->Step(u, t, dt);
   }

   u_gf.SetFromTrueDofs(u);

   mfem::ConstantCoefficient zero(0.0);

   double u_norm = u_gf.ComputeLpError(2.0, zero);

   EXPECT_NEAR(2.53521, u_norm, 0.00001);

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
  if (!args.Good())
  {
    if (myid == 0)
    {
      args.PrintUsage(std::cout);
    }
    MPI_Finalize();
    return 1;
  }
  if (myid == 0)
  {
    args.PrintOptions(std::cout);
  }

  result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}

double InitialTemperature(const mfem::Vector &x)
{
   if (x.Norml2() < 0.5)
   {
      return 2.0;
   }
   else
   {
      return 1.0;
   }
}

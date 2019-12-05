// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause) 

#include <gtest/gtest.h>

#include "mfem.hpp"
#include "solvers/quasistatic_solver.hpp"
#include <fstream>

TEST(quasistatic_solver, qs_solve)
{
   MPI_Barrier(MPI_COMM_WORLD);

   // mesh
   const char *mesh_file = "../../data/beam-hex.mesh";

   // Open the mesh
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

   x_inc=0.0;
   
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
   traction(1) = 3.0e-4;

   mfem::VectorConstantCoefficient traction_coef(traction);
   
   // construct the nonlinear mechanics operator
   QuasistaticSolver oper(fe_space, ess_bdr, trac_bdr,
                          0.25, 5.0, traction_coef,
                          1.0e-2, 1.0e-4, 
                          500, true, false);
   
   // declare incremental nodal displacement solution vector
   mfem::Vector x_sol(fe_space.TrueVSize());
   x_inc.GetTrueDofs(x_sol);

   // Solve the Newton system 
   int converged = oper.Solve(x_sol);     

   // distribute the solution vector to x_cur
   x_inc.Distribute(x_sol);
 
   mfem::Vector zero(dim);
   zero = 0.0;
   mfem::VectorConstantCoefficient zerovec(zero);

   double x_norm = x_inc.ComputeLpError(2.0, zerovec); 
   
   EXPECT_NEAR(0.770937, x_norm, 0.00001);
   EXPECT_EQ(converged, 1);
   
   delete pmesh;
   
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

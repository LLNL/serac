// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause) 

#include <gtest/gtest.h>

#include "mfem.hpp"
#include "solvers/hyperelastic_solver.hpp"
#include <fstream>

using namespace std;
using namespace mfem;

TEST(cg_convection_solver, cg_conv_solve)
{
   MPI_Barrier(MPI_COMM_WORLD);

   // mesh
   const char *mesh_file = "../../data/beam-hex.mesh";

   // Open the mesh
   ifstream imesh(mesh_file);
   Mesh* mesh = new Mesh(imesh, 1, 1, true);
   imesh.close();

   // declare pointer to parallel mesh object
   ParMesh *pmesh = NULL;
   mesh->UniformRefinement();
   
   pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   int dim = pmesh->Dimension();
   
   // Define the finite element spaces for displacement field
   H1_FECollection fe_coll(1, dim);
   ParFiniteElementSpace fe_space(pmesh, &fe_coll, dim);

   // Define a grid function for the global reference configuration, the beginning 
   // step configuration, the global deformation, the current configuration/solution 
   // guess, and the incremental nodal displacements
   ParGridFunction x_inc(&fe_space);   

   x_inc=0.0;
   
   // define a boundary attribute array and initialize to 0
   Array<int> ess_bdr;
   ess_bdr.SetSize(fe_space.GetMesh()->bdr_attributes.Max());
   ess_bdr = 0;

   // boundary attribute 1 (index 0) is fixed (Dirichlet)
   ess_bdr[0] = 1;
   
   Array<int> trac_bdr;   
   trac_bdr.SetSize(fe_space.GetMesh()->bdr_attributes.Max());
      
   trac_bdr = 0;
   trac_bdr[1] = 1;
   
   // define the traction vector
   Vector traction(dim);
   traction = 0.0;
   traction(1) = 1.0e-4;

   VectorConstantCoefficient traction_coef(traction);
   
   // construct the nonlinear mechanics operator
   NonlinearMechOperator oper(fe_space, ess_bdr, trac_bdr,
                              0.25, 5.0, traction_coef,
                              1.0e-2, 1.0e-4, 
                              500, true, false);
   
   // declare incremental nodal displacement solution vector
   Vector x_sol(fe_space.TrueVSize());
   x_inc.GetTrueDofs(x_sol);

   // Solve the Newton system 
   int converged = oper.Solve(x_sol);     

   ASSERT_EQ(converged, 1);
   
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

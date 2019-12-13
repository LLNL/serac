// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause) 

#include <gtest/gtest.h>

#include "mfem.hpp"
#include "solvers/linear_elasticity_solver.hpp"
#include <fstream>

TEST(linearelastic_solver, le_solve)
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
   mfem::ParGridFunction x(&fe_space);   

   x=0.0;
   
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

   mfem::VectorArrayCoefficient f(dim);
   for (int i = 0; i < dim-1; i++)
   {
      f.Set(i, new mfem::ConstantCoefficient(0.0));
   }
   {
      mfem::Vector pull_force(pmesh->bdr_attributes.Max());
      pull_force = 0.0;
      pull_force(1) = -1.0e-4;
      f.Set(dim-1, new mfem::PWConstCoefficient(pull_force));
   }


   // define the traction vector
   mfem::Vector traction(dim);
   traction = 0.0;
   traction(1) = 1.0e-4;

   mfem::VectorConstantCoefficient traction_coef(traction);
   mfem::ConstantCoefficient mu_coef(0.25);
   mfem::ConstantCoefficient K_coef(5.0);


   // construct the nonlinear mechanics operator
   LinearElasticSolver oper(fe_space, ess_bdr, trac_bdr,
                            mu_coef, K_coef, f,
                            1.0e-4, 1.0e-10, 
                            500, true, false);
   
  // Solve the Newton system 
   bool converged = oper.Solve(x); 

   {
      int myid;
      MPI_Comm_rank(MPI_COMM_WORLD, &myid);


      mfem::GridFunction *nodes = pmesh->GetNodes();
      *nodes += x;
      x *= -1;

      std::ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << std::setfill('0') << std::setw(6) << myid;
      sol_name << "sol." << std::setfill('0') << std::setw(6) << myid;

      std::ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      std::ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

 
   mfem::Vector zero(dim);
   zero = 0.0;
   mfem::VectorConstantCoefficient zerovec(zero);

   double x_norm = x.ComputeLpError(2.0, zerovec); 
   
   EXPECT_NEAR(0.770937, x_norm, 0.00001);
   EXPECT_TRUE(converged);
   
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

//***********************************************************************
//
//   SERAC - Nonlinear Implicit Contact Proxy App
//
//   Description: The purpose of this code is to act as a proxy app
//                for nonlinear implicit mechanics codes at LLNL. This
//                initial version is copied from a previous version
//                of the ExaConsist AM miniapp.
//
//
//***********************************************************************

#include "mfem.hpp"
#include "solvers/hyperelastic_solver.hpp"
#include "coefficients/loading_functions.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
   // print the version of the code being run
   printf("MFEM Version: %d \n", GetVersion());

   // Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // mesh
   const char *mesh_file = "../../data/beam-hex.mesh";

   // serial and parallel refinement levels
   int ser_ref_levels = 0;
   int par_ref_levels = 0;

   // polynomial interpolation order
   int order = 1;

   // newton input args
   double newton_rel_tol = 1.0e-2;
   double newton_abs_tol = 1.0e-4;
   int newton_iter = 500;
   
   // solver input args
   bool gmres_solver = true;
   bool slu_solver = false;

   // neo-Hookean material parameters
   double mu = 0.25;
   double K = 5.0;

   // loading parameters
   double tx = 0.0;
   double ty = 1.0e-3;
   double tz = 0.0;   
   
   // specify all input arguments
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&mu, "-mu", "--shear-modulus",
                  "Shear modulus in the Neo-Hookean hyperelastic model.");
   args.AddOption(&K, "-K", "--bulk-modulus",
                  "Bulk modulus in the Neo-Hookean hyperelastic model.");
   args.AddOption(&tx, "-tx", "--traction-x",
                  "Cantilever tip traction in the x direction.");
   args.AddOption(&ty, "-ty", "--traction-y",
                  "Cantilever tip traction in the y direction.");
   args.AddOption(&tz, "-tz", "--traction-z",
                  "Cantilever tip traction in the z direction.");
   args.AddOption(&slu_solver, "-slu", "--superlu", "-no-slu",
                  "--no-superlu", "Use the SuperLU Solver.");
   args.AddOption(&gmres_solver, "-gmres", "--gmres", "-no-gmres", "--no-gmres",
                   "Use gmres, otherwise minimum residual is used.");
   args.AddOption(&newton_rel_tol, "-rel", "--relative-tolerance",
                  "Relative tolerance for the Newton solve.");
   args.AddOption(&newton_abs_tol, "-abs", "--absolute-tolerance",
                  "Absolute tolerance for the Newton solve.");
   args.AddOption(&newton_iter, "-it", "--newton-iterations",
                  "Maximum iterations for the Newton solve.");

   // Parse the arguments and check if they are good
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // Open the mesh
   Mesh *mesh;
   ifstream imesh(mesh_file);
   if (!imesh)
      {
         if (myid == 0)
            {
               cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
            }
         MPI_Finalize();
         return 2;
      }
  
   mesh = new Mesh(imesh, 1, 1, true);
   imesh.close();

   // declare pointer to parallel mesh object
   ParMesh *pmesh = NULL;
   
   // mesh refinement if specified in input
   for (int lev = 0; lev < ser_ref_levels; lev++)
      {
         mesh->UniformRefinement();
      }

   pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   for (int lev = 0; lev < par_ref_levels; lev++)
      {
         pmesh->UniformRefinement();
      }

   delete mesh;

   int dim = pmesh->Dimension();
   
   // Define the finite element spaces for displacement field
   H1_FECollection fe_coll(order, dim);
   ParFiniteElementSpace fe_space(pmesh, &fe_coll, dim);

   HYPRE_Int glob_size = fe_space.GlobalTrueVSize();

   // Print the mesh statistics
   if (myid == 0)
   {
      std::cout << "***********************************************************\n";
      std::cout << "dim(u) = " << glob_size << "\n";
      std::cout << "***********************************************************\n";
   }   

   // Define a grid function for the global reference configuration, the beginning 
   // step configuration, the global deformation, the current configuration/solution 
   // guess, and the incremental nodal displacements
   ParGridFunction x_ref(&fe_space);
   ParGridFunction x_cur(&fe_space);
   ParGridFunction x_fin(&fe_space);
   ParGridFunction x_inc(&fe_space);   

   // Project the initial and reference configuration functions onto the appropriate grid functions
   VectorFunctionCoefficient deform(dim, InitialDeformation);
   VectorFunctionCoefficient refconfig(dim, ReferenceConfiguration);
   
   // Initialize the reference and beginning step configuration grid functions 
   // with the refconfig vector function coefficient.
   x_ref.ProjectCoefficient(refconfig);

   // initialize x_cur, boundary condition, deformation, and 
   // incremental nodal displacment grid functions by projection the 
   // VectorFunctionCoefficient function onto them
   x_cur.ProjectCoefficient(deform);

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
   traction(0) = tx;
   traction(1) = ty;   
   if (dim == 3) {
      traction(2) = tz;
   }

   VectorConstantCoefficient traction_coef(traction);
   
   // construct the nonlinear mechanics operator
   NonlinearMechOperator oper(fe_space, ess_bdr, trac_bdr,
                              mu, K, traction_coef,
                              newton_rel_tol, newton_abs_tol, 
                              newton_iter, gmres_solver, slu_solver);
   
   // declare incremental nodal displacement solution vector
   Vector x_sol(fe_space.TrueVSize());
   x_cur.GetTrueDofs(x_sol);

   // Solve the Newton system 
   oper.Solve(x_sol);     

   // distribute the solution vector to x_cur
   x_fin.Distribute(x_sol);

   // Save the displaced mesh. These are snapshots of the endstep current 
   // configuration. Later add functionality to not save the mesh at each timestep.

   GridFunction *nodes = &x_fin; // set a nodes grid function to global current configuration
   int owns_nodes = 0;
   pmesh->SwapNodes(nodes, owns_nodes); // pmesh has current configuration nodes

   ostringstream mesh_name, pressure_name, deformation_name;
   mesh_name << "mesh." << setfill('0') << setw(6) << myid;
   deformation_name << "deformation." << setfill('0') << setw(6) << myid;

   ofstream mesh_ofs(mesh_name.str().c_str());
   mesh_ofs.precision(8);
   pmesh->Print(mesh_ofs);
    
   ofstream deformation_ofs(deformation_name.str().c_str());
   deformation_ofs.precision(8);

   subtract(x_fin, x_ref, x_inc);
   x_inc.Save(deformation_ofs);
      
   // Free the used memory.
   delete pmesh;

   MPI_Finalize();

   return 0;
}


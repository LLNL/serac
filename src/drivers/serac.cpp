
#include "mfem.hpp"
#include "solvers/cg_diffusion_solver.hpp"
#include "solvers/cg_convection_solver.hpp"
#include "solvers/dg_diffusion_solver.hpp"
#include "solvers/dg_convection_solver.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;

int main(int argc, char *argv[])
{
     
   // Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int ser_ref_levels = 1;
   int par_ref_levels = 0;
   int order = 1;
   double sigma = -1.0;
   double kappa = -1.0;
   double svel = 5.0;
   
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial,"
                  " -1 for auto.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&sigma, "-s", "--sigma",
                  "One of the two DG penalty parameters, typically +1/-1."
                  " See the documentation of class DGDiffusionIntegrator.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "One of the two DG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");
   args.AddOption(&svel, "-v", "--velocity",
                  "Background velocity.");
   
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
   if (kappa < 0)
   {
      kappa = (order+1)*(order+1);
   }
   if (myid == 0)
   {
      args.PrintOptions(std::cout);
   }

   //    Read the (serial) mesh from the given mesh file on all processors. We
   //    can handle triangular, quadrilateral, tetrahedral and hexahedral meshes
   //    with the same code. NURBS meshes are projected to second order meshes.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);

   //    Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ser_ref_levels' of uniform refinement. By default,
   //    or if ser_ref_levels < 0, we choose it to be the largest number that
   //    gives a final mesh with no more than 50,000 elements.
   {
      for (int l = 0; l < ser_ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }
   if (mesh->NURBSext)
   {
      mesh->SetCurvature(std::max(order, 1));
   }

   //    Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   // Construct the CG and DG diffusion solvers
   CGDiffusionSolver cg_diffusion_solver(pmesh, order);
   DGDiffusionSolver dg_diffusion_solver(pmesh, order, kappa, sigma);   

   // Set the problem coefficients for the diffusion solvers
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   ConstantCoefficient diff(1.0);
   
   cg_diffusion_solver.SetSource(one);
   cg_diffusion_solver.SetDiffusionCoefficient(diff);

   dg_diffusion_solver.SetSource(one);
   dg_diffusion_solver.SetDiffusionCoefficient(diff);

   // Mark all boundaries as Dirichlet with value of 1
   Array<int> dof_markers(pmesh->bdr_attributes.Max());
   dof_markers = 1;
   cg_diffusion_solver.SetDirichletBoundary(dof_markers, one);
   dg_diffusion_solver.SetDirichletBoundary(dof_markers, one);   

   // Solve the diffusion system using CG and DG
   cg_diffusion_solver.Solve();
   dg_diffusion_solver.Solve();   

   // Get the solution fields
   ParGridFunction *cg_diffusion = cg_diffusion_solver.GetSolutionField();
   ParGridFunction *dg_diffusion = dg_diffusion_solver.GetSolutionField();   

   // Construct the CG and DG convection solvers
   CGConvectionSolver cg_convection_solver(pmesh, order);
   DGConvectionSolver dg_convection_solver(pmesh, order, kappa, sigma);   

   // Set the problem coefficients for the convection solvers
   cg_convection_solver.SetSource(one);
   cg_convection_solver.SetDiffusionCoefficient(diff);

   dg_convection_solver.SetSource(one);
   dg_convection_solver.SetDiffusionCoefficient(diff);
   
   Vector vel(pmesh->Dimension());
   vel = 0.0;
   vel(0) = svel;
   
   VectorConstantCoefficient velocity(vel);
   cg_convection_solver.SetConvectionCoefficient(velocity);
   dg_convection_solver.SetConvectionCoefficient(velocity);   

   // Mark all boundaries as inflow with value of 1
   dof_markers = 1;
   cg_convection_solver.SetInflowBoundary(dof_markers, one);
   dg_convection_solver.SetInflowBoundary(dof_markers, one);   

   // Solve the systems
   cg_convection_solver.Solve();
   dg_convection_solver.Solve();   

   // Get the solution fields
   ParGridFunction *cg_convection = cg_convection_solver.GetSolutionField();
   ParGridFunction *dg_convection = dg_convection_solver.GetSolutionField();   

   // Output the results in visit format
   VisItDataCollection visit_dc("Flobat-Convection", pmesh);

   visit_dc.RegisterField("cg-diffusion", cg_diffusion);
   visit_dc.RegisterField("cg-convection", cg_convection);   
   visit_dc.RegisterField("dg-diffusion", dg_diffusion);
   visit_dc.RegisterField("dg-convection", dg_convection);   
   visit_dc.SetCycle(0);
   visit_dc.SetTime(0.0);
   visit_dc.Save();

   delete pmesh;
   
   MPI_Finalize();

   return 0;
}

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
#include "mechanics_coefficient.hpp"
#include "mechanics_integrators.hpp"
#include "mechanics_solver.hpp"
#include "BCData.hpp"
#include "BCManager.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;

class SimVars
{
protected:
   double time;
   double dt;
public:
   double GetTime() { return time; }
   double GetDTime() { return dt; }

   void SetTime(double t) { time = t; }
   void SetDt(double dtime) { dt = dtime; }
};

class NonlinearMechOperator : public TimeDependentOperator
{
public:
   SimVars solVars;
protected:
   ParFiniteElementSpace &fe_space;

   ParNonlinearForm *Hform;
   mutable Operator *Jacobian;
   const Vector *x;

   /// Newton solver for the operator
   MechNewtonSolver newton_solver;
   /// Solver for the Jacobian solve in the Newton method
   Solver *J_solver;
   /// Preconditioner for the Jacobian
   Solver *J_prec;
   /// nonlinear model 
   MechModel *model;

public:
   NonlinearMechOperator(ParFiniteElementSpace &fes,
                         Array<int> &ess_bdr,
                         double rel_tol,
                         double abs_tol,
                         int iter,
                         bool gmres,
                         bool slu)

   /// Required to use the native newton solver
   virtual Operator &GetGradient(const Vector &x) const;
   virtual void Mult(const Vector &k, Vector &y) const;

   /// Driver for the newton solver
   void Solve(Vector &x) const;

   /// Get essential true dof list, if required
   const Array<int> &GetEssTDofList();

   /// Get FE space
   const ParFiniteElementSpace *GetFESpace() { return &fe_space; }

   void SetTime(const double t);
   void SetDt(const double dt);

   virtual ~NonlinearMechOperator();
};

void visualize(ostream &out, ParMesh *mesh, ParGridFunction *deformed_nodes,
               ParGridFunction *field, const char *field_name = NULL,
               bool init_vis = false);

// set kinematic functions and boundary condition functions
void ReferenceConfiguration(const Vector &x, Vector &y);
void InitialDeformation(const Vector &x, double t,  Vector &y);
void Velocity(const Vector &x, double t, Vector &y);
void DirBdrFunc(const Vector &x, double t, int attr_id, Vector &y);
void InitGridFunction(const Vector &x, Vector &y);

// set the time step on the boundary condition objects
void setBCTimeStep(double dt, int nDBC);

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

   // final simulation time and time step (set each to 1.0 for 
   // single step debug)
   double t_final = 1.0;
   double dt = 1.0;

   // visualization input args
   bool visualization = true;
   int vis_steps = 1;

   // newton input args
   double newton_rel_tol = 1.0e-6;
   double newton_abs_tol = 1.0e-8;
   int newton_iter = 500;
   
   // solver input args
   bool gmres_solver = true;
   bool slu_solver = false;

   // boundary condition input args
   Array<int> ess_id;   // essential bc ids for the whole boundary
   Vector     ess_disp; // vector of displacement components for each attribute in ess_id
   Array<int> ess_comp; // component combo (x,y,z = -1, x = 1, y = 2, z = 3, 
                        // xy = 4, yz = 5, xz = 6, free = 0 

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
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&slu_solver, "-slu", "--superlu", "-no-slu",
                  "--no-superlu", "Use the SuperLU Solver.");
   args.AddOption(&gmres_solver, "-gmres", "--gmres", "-no-gmres", "--no-gmres",
                   "Use gmres, otherwise minimum residual is used.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&newton_rel_tol, "-rel", "--relative-tolerance",
                  "Relative tolerance for the Newton solve.");
   args.AddOption(&newton_abs_tol, "-abs", "--absolute-tolerance",
                  "Absolute tolerance for the Newton solve.");
   args.AddOption(&newton_iter, "-it", "--newton-iterations",
                  "Maximum iterations for the Newton solve.");
   args.AddOption(&ess_id, "-attrid", "--dirichlet-attribute-ids",
                  "Attribute IDs for dirichlet boundary conditions.");
   args.AddOption(&ess_disp, "-disp", "--dirichlet-disp", 
                  "Final (x,y,z) displacement components for each dirichlet BC.");
   args.AddOption(&ess_comp, "-bcid", "--bc-comp-id",
                  "Component ID for essential BCs.");

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

   // Define the finite element space for the stress field
   RT_FECollection hdiv_fe_coll(order, dim);
   ParFiniteElementSpace hdiv_fe_space(pmesh, &hdiv_fe_coll, dim);

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
   ParGridFunction x_beg(&fe_space);
   ParGridFunction x_def(&fe_space);
   ParGridFunction x_cur(&fe_space);
   ParGridFunction x_inc(&fe_space);

   // define a vector function coefficient for the initial deformation 
   // (based on a velocity projection) and reference configuration.
   // Additionally define a vector function coefficient for computing 
   // the grid velocity prior to a velocity projection
   VectorFunctionCoefficient velProj(dim, InitialDeformation);
   VectorFunctionCoefficient refconfig(dim, ReferenceConfiguration);
   VectorFunctionCoefficient compVel(dim, Velocity);
  
   // Initialize the reference and beginning step configuration grid functions 
   // with the refconfig vector function coefficient.
   x_beg.ProjectCoefficient(refconfig);
   x_ref.ProjectCoefficient(refconfig);

   // Define grid function for the nodal displacement solution grid function
   // WITH Dirichlet BCs
   ParGridFunction x_bar(&fe_space);

   // Define a VectorFunctionCoefficient to initialize a grid function
   VectorFunctionCoefficient init_grid_func(dim, InitGridFunction);

   // initialize x_cur, boundary condition, deformation, and 
   // incremental nodal displacment grid functions by projection the 
   // VectorFunctionCoefficient function onto them
   x_cur.ProjectCoefficient(init_grid_func);
   x_bar.ProjectCoefficient(init_grid_func);
   x_def.ProjectCoefficient(init_grid_func);
   x_inc.ProjectCoefficient(init_grid_func);

   // define a boundary attribute array and initialize to 0
   Array<int> ess_bdr;
   ess_bdr.SetSize(fe_space.GetMesh()->bdr_attributes.Max());
   ess_bdr = 0;

   // setup inhomogeneous essential boundary conditions using the boundary 
   // condition manager (BCManager) and boundary condition data (BCData) 
   // classes developed for ExaConstit.
   if (ess_disp.Size() != 3*ess_id.Size()) {
      cerr << "\nMust specify three Dirichlet components per essential boundary attribute" << '\n' << endl;
   }

   int numDirBCs = 0;
   for (int i=0; i<ess_id.Size(); ++i) {
      // set the boundary condition id based on the attribute id
      int bcID = ess_id[i];

      // instantiate a boundary condition manager instance and 
      // create a BCData object
      BCManager & bcManager = BCManager::getInstance();
      BCData & bc = bcManager.CreateBCs( bcID );

      // set the displacement component values
      bc.essDisp[0] = ess_disp[3*i];
      bc.essDisp[1] = ess_disp[3*i+1];
      bc.essDisp[2] = ess_disp[3*i+2];
      bc.compID = ess_comp[i];

      // set the final simulation time 
      bc.tf = t_final;

      // set the boundary condition scales
      bc.setScales();

      // set the active boundary attributes
      if (bc.compID != 0) {
         ess_bdr[i] = 1;
      }
      ++numDirBCs;
   }

   // declare a VectorFunctionRestrictedCoefficient over the boundaries that have attributes
   // associated with a Dirichlet boundary condition (ids provided in input)
   VectorFunctionRestrictedCoefficient ess_bdr_func(dim, DirBdrFunc, ess_bdr);

   // Construct the nonlinear mechanics operator. Note that q_grain0 is
   // being passed as the matVars0 quadarture function. This is the only 
   // history variable considered at this moment. Consider generalizing 
   // this where the grain info is a possible subset only of some 
   // material history variable quadrature function. Also handle the 
   // case where there is no grain data.
   NonlinearMechOperator oper(fe_space, ess_bdr, 
                              newton_rel_tol, newton_abs_tol, 
                              newton_iter, gmres_solver, slu_solver);

   // get the essential true dof list. This may not be used.
   const Array<int> ess_tdof_list = oper.GetEssTDofList();
   
   // declare incremental nodal displacement solution vector
   Vector x_sol(fe_space.TrueVSize());
   x_sol = 0.0;

   // initialize visualization if requested 
   socketstream vis_u, vis_p;
   if (visualization) {
      char vishost[] = "localhost";
      int  visport   = 19916;
      vis_u.open(vishost, visport);
      vis_u.precision(8);
      visualize(vis_u, pmesh, &x_beg, &x_def, "Deformation", true);
      // Make sure all ranks have sent their 'u' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());
   }

   // initialize/set the time
   double t = 0.0;
   oper.SetTime(t); 

   bool last_step = false;

   // enter the time step loop. This was modeled after example 10p.
   for (int ti = 1; !last_step; ti++)
   {

      printf("time: %d \n", ti);
      // compute time step (this calculation is pulled from ex10p.cpp)
      double dt_real = min(dt, t_final - t);

      // set the time step on the boundary conditions
      setBCTimeStep(dt_real, numDirBCs);

      // compute current time
      t = t + dt_real;

      // set time on the simulation variables and the model through the 
      // nonlinear mechanics operator class
      oper.SetTime(t);
      oper.SetDt(dt_real);

      // set the time for the nonzero Dirichlet BC function evaluation
      //non_zero_ess_func.SetTime(t);
      ess_bdr_func.SetTime(t);

      // register Dirichlet BCs. 
      ess_bdr = 1;

      // overwrite entries in x_bar for dofs with Dirichlet 
      // boundary conditions (note, this routine overwrites, not adds).
      // Note: these are prescribed incremental nodal displacements at 
      // Dirichlet BC dofs.
      x_bar.ProjectBdrCoefficient(ess_bdr_func); // don't need attr list as input
                                                 // pulled off the 
                                                 // VectorFunctionRestrictedCoefficient

      // add the displacement bcs with velocity projection to the beginning step 
      // configuration as full nodal grid function used to populate solution vector
      add(x_beg, x_bar, x_cur);

      // populate the solution vector, x_sol, with the true dofs entries in x_cur.
      // At this point we initialized x_bar, performed a velocity projection for 
      // all dofs, and then over-wrote the Dirichlet BC dofs with the boundary condition 
      // function.
      x_cur.GetTrueDofs(x_sol);


      // Solve the Newton system 
      oper.Solve(x_sol);     
      oper.UpdateModel(x_sol);

      // distribute the solution vector to x_cur
      x_cur.Distribute(x_sol);

      // set the incremental nodal displacement grid function
      subtract(x_cur, x_beg, x_inc);

      // set the end step deformation wrt global reference configuration
      subtract(x_cur, x_ref, x_def);

      // update beginning step solution
      x_beg = x_cur;

      last_step = (t >= t_final - 1e-8*dt);

      if (last_step || (ti % vis_steps) == 0)
      {
         if (myid == 0)
         {
            cout << "step " << ti << ", t = " << t << endl;
         }
      }

      { 
         // Save the displaced mesh. These are snapshots of the endstep current 
         // configuration. Later add functionality to not save the mesh at each timestep.

         GridFunction *nodes = &x_beg; // set a nodes grid function to global current configuration
         int owns_nodes = 0;
         pmesh->SwapNodes(nodes, owns_nodes); // pmesh has current configuration nodes

         ostringstream mesh_name, deformed_name;
         mesh_name << "mesh." << setfill('0') << setw(6) << myid << "_" << ti << ".vtk";

         // saving mesh for plotting. pmesh has global current configuration nodal coordinates
         ofstream mesh_ofs(mesh_name.str().c_str());
         mesh_ofs.precision(8);
         pmesh->PrintVTK(mesh_ofs); 
       

      } // end output scope

   } // end loop over time steps
      
   // Free the used memory.
   delete pmesh;

   MPI_Finalize();

   return 0;
}

NonlinearMechOperator::NonlinearMechOperator(ParFiniteElementSpace &fes,
                                             Array<int> &ess_bdr,
                                             double rel_tol,
                                             double abs_tol,
                                             int iter,
                                             bool gmres,
                                             bool slu)
   : TimeDependentOperator(fes.TrueVSize()), fe_space(fes),
     newton_solver(fes.GetComm())
{
   Vector * rhs;
   rhs = NULL;

   // Define the parallel nonlinear form 
   Hform = new ParNonlinearForm(&fes);

   // Set the essential boundary conditions
   Hform->SetEssentialBCPartial(ess_bdr, rhs); 

   model = new NeoHookean(&q_sigma0, &q_sigma1, &q_matGrad, &q_matVars0, 
                          &q_matVars1, &q_kinVars0, matProps, numProps, nStateVars,
                          80.E3, 140.E3, 1.0);

   // Add the hyperelastic integrator
   Hform->AddDomainIntegrator(new ExaNLFIntegrator(dynamic_cast<NeoHookean*>(model)));

   if (gmres) {
      HypreBoomerAMG *prec_amg = new HypreBoomerAMG();
      prec_amg->SetPrintLevel(0);
      prec_amg->SetElasticityOptions(&fe_space);
      J_prec = prec_amg;

      GMRESSolver *J_gmres = new GMRESSolver(fe_space.GetComm());
      J_gmres->SetRelTol(rel_tol);
      J_gmres->SetAbsTol(1e-12);
      J_gmres->SetMaxIter(300);
      J_gmres->SetPrintLevel(0);
      J_gmres->SetPreconditioner(*J_prec);
      J_solver = J_gmres; 

   } 
   // retain super LU solver capabilities
   else if (slu) { 
      SuperLUSolver *superlu = NULL;
      superlu = new SuperLUSolver(MPI_COMM_WORLD);
      superlu->SetPrintStatistics(false);
      superlu->SetSymmetricPattern(false);
      superlu->SetColumnPermutation(superlu::PARMETIS);
      
      J_solver = superlu;
      J_prec = NULL;
   }
   else {
      printf("using minres solver \n");
      HypreSmoother *J_hypreSmoother = new HypreSmoother;
      J_hypreSmoother->SetType(HypreSmoother::l1Jacobi);
      J_hypreSmoother->SetPositiveDiagonal(true);
      J_prec = J_hypreSmoother;

      MINRESSolver *J_minres = new MINRESSolver(fe_space.GetComm());
      J_minres->SetRelTol(rel_tol);
      J_minres->SetAbsTol(0.0);
      J_minres->SetMaxIter(300);
      J_minres->SetPrintLevel(-1);
      J_minres->SetPreconditioner(*J_prec);
      J_solver = J_minres;

   }

   // Set the newton solve parameters
   newton_solver.iterative_mode = true;
   newton_solver.SetSolver(*J_solver);
   newton_solver.SetOperator(*this);
   newton_solver.SetPrintLevel(1); 
   newton_solver.SetRelTol(rel_tol);
   newton_solver.SetAbsTol(abs_tol);
   newton_solver.SetMaxIter(iter);
}

const Array<int> &NonlinearMechOperator::GetEssTDofList()
{
   return Hform->GetEssentialTrueDofs();
}

// Solve the Newton system
void NonlinearMechOperator::Solve(Vector &x) const
{
   Vector zero;
   newton_solver.Mult(zero, x);

   MFEM_VERIFY(newton_solver.GetConverged(), "Newton Solver did not converge.");
}

// compute: y = H(x,p)
void NonlinearMechOperator::Mult(const Vector &k, Vector &y) const
{
   // Apply the nonlinear form
   Hform->Mult(k, y);
}

// Compute the Jacobian from the nonlinear form
Operator &NonlinearMechOperator::GetGradient(const Vector &x) const
{
   Jacobian = &Hform->GetGradient(x);
   return *Jacobian;
}

void NonlinearMechOperator::SetTime(const double t)
{
   solVars.SetTime(t);
   model->SetModelTime(t);
   return;
}

void NonlinearMechOperator::SetDt(const double dt)
{
   solVars.SetDt(dt);
   model->SetModelDt(dt);
   return;
}

NonlinearMechOperator::~NonlinearMechOperator()
{
   delete J_solver;
   if (J_prec != NULL) {
      delete J_prec;
   }
   delete model;
}

// In line visualization
void visualize(ostream &out, ParMesh *mesh, ParGridFunction *deformed_nodes,
               ParGridFunction *field, const char *field_name, bool init_vis)
{
   if (!out)
   {  
      return;
   }

   GridFunction *nodes = deformed_nodes;
   int owns_nodes = 0;

   mesh->SwapNodes(nodes, owns_nodes);

   out << "parallel " << mesh->GetNRanks() << " " << mesh->GetMyRank() << "\n";
   out << "solution\n" << *mesh << *field;

   mesh->SwapNodes(nodes, owns_nodes);

   if (init_vis)
   {
      out << "window_size 800 800\n";
      out << "window_title '" << field_name << "'\n";
      if (mesh->SpaceDimension() == 2)
      {
         out << "view 0 0\n"; // view from top
         out << "keys jl\n";  // turn off perspective and light
      }
      out << "keys cm\n";         // show colorbar and mesh
      out << "autoscale value\n"; // update value-range; keep mesh-extents fixed
      out << "pause\n";
   }
   out << flush;
}

void ReferenceConfiguration(const Vector &x, Vector &y)
{
   // set the reference, stress free, configuration
   y = x;
}


void InitialDeformation(const Vector &x, double t,  Vector &y)
{
   // this performs a velocity projection 
   
   // Note: x comes in initialized to 0.0 on the first time step, 
   // otherwise it is coming in as incremental nodal velocity, which is 
   // the previous step's incremental nodal displacement solution 
   // divided by the previous time step

   // get the time step off the boundary condition manager
   // for the first BC, which there has to be at least one of
   BCManager & bcManager = BCManager::getInstance();
   BCData & bc_data = bcManager.GetBCInstance(1);

   double dt = bc_data.dt;

   // velocity projection is the last delta_x solution (x_cur) times 
   // the current timestep.
   Vector temp_x(x);
   temp_x *= dt;
   y = temp_x;
}

void Velocity(const Vector &x, double t, Vector &y)
{
   BCManager & bcManager = BCManager::getInstance();
   BCData & bc_data = bcManager.GetBCInstance(1);
 
   double dt = bc_data.dt;

   // compute the grid velocity by dividing by dt
   Vector temp_x = x;
   temp_x /= dt;
   y = temp_x; 
}

void DirBdrFunc(const Vector &x, double t, int attr_id, Vector &y)
{
   BCManager & bcManager = BCManager::getInstance();
   BCData & bc = bcManager.GetBCInstance(attr_id);

   bc.setDirBCs(x, t, y);
}

void InitGridFunction(const Vector &x, Vector &y)
{
   y = 0.;
}

void setBCTimeStep(double dt, int nDBC)
{
   for (int i=0; i<nDBC; ++i) {
      BCManager & bcManager = BCManager::getInstance();
      BCData & bc = bcManager.CreateBCs( i+1 );
      bc.dt = dt;
   }

}


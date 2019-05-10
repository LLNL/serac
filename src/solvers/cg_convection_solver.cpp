#include "cg_convection_solver.hpp"

using namespace mfem;

CGConvectionSolver::CGConvectionSolver(ParMesh *pmesh, int order) :
   m_pmesh(pmesh), m_order(order), m_source(nullptr), m_diffusion(nullptr), m_convection(nullptr), m_outflow_values(nullptr),
   m_inflow_values(nullptr)
{
   m_fe_coll = new H1_FECollection(m_order, m_pmesh->Dimension());
   m_fe_space = new ParFiniteElementSpace(m_pmesh, m_fe_coll);
   m_sol = new ParGridFunction(m_fe_space);
}

void CGConvectionSolver::SetSource(Coefficient & source)
{
   m_source = &source;
}

void CGConvectionSolver::SetDiffusionCoefficient(Coefficient & diff)
{
   m_diffusion = &diff;
}

void CGConvectionSolver::SetConvectionCoefficient(VectorCoefficient & conv)
{
   m_convection = &conv;
}

void CGConvectionSolver::SetOutflowBoundary(Array<int> &outflow_indicator, Coefficient & outflow_values)
{
   m_outflow_indicator = outflow_indicator;
   m_outflow_values = &outflow_values;
}

void CGConvectionSolver::SetInflowBoundary(Array<int> &inflow_indicator, Coefficient & inflow_values)
{
   m_inflow_indicator = inflow_indicator;
   m_inflow_values = &inflow_values;
}

int CGConvectionSolver::Solve()
{
   *m_sol = 0.0;
   Array<int> ess_tdof_list;

   // Create the linear form RHS object
   ParLinearForm *b = new ParLinearForm(m_fe_space);
   if (m_source != nullptr) {
      // add the volume source if defined
      b->AddDomainIntegrator(new DomainLFIntegrator(*m_source));
   }

   if (m_inflow_values != nullptr) {
      // Get the essential dof numbers 
      m_fe_space->GetEssentialTrueDofs(m_inflow_indicator, ess_tdof_list);
      // Project the boundary values onto the essential dofs
      m_sol->ProjectBdrCoefficient(*m_inflow_values, m_inflow_indicator);
   }

   if (m_outflow_values != nullptr) {
      // Add the outflow boundary integrator
      b->AddBoundaryIntegrator(new BoundaryLFIntegrator(*m_outflow_values), m_outflow_indicator);
   }

   // Assemble the linear form object
   b->Assemble();

   // Create the bilinear form object
   ParBilinearForm *a = new ParBilinearForm(m_fe_space);

   // Add the CG diffusion and convection integrators
   a->AddDomainIntegrator(new DiffusionIntegrator(*m_diffusion));
   a->AddDomainIntegrator(new ConvectionIntegrator(*m_convection));

   // Assemble the bilinear form object
   a->Assemble();

   // Create the linear algebra objects for the solve
   HypreParMatrix *A = new HypreParMatrix;
   Vector B, X;

   // Form the linear algebra system from the abstract objects and apply boundary conditions
   a->FormLinearSystem(ess_tdof_list, *m_sol, *b, *A, X, B);

   // Setup the solver parameters
   HypreBoomerAMG *amg = new HypreBoomerAMG(*A);
   amg->SetPrintLevel(0);
   GMRESSolver gmres(MPI_COMM_WORLD);
   gmres.SetAbsTol(0.0);
   gmres.SetRelTol(1e-12);
   gmres.SetMaxIter(200);
   gmres.SetKDim(10);
   gmres.SetPrintLevel(0);
   gmres.SetOperator(*A);
   gmres.SetPreconditioner(*amg);

   // Solve the system and test for convergence
   gmres.Mult(B, X);
   int converged = gmres.GetConverged();

   // Communicate the shared nodes
   a->RecoverFEMSolution(X, *b, *m_sol);      
   
   if (converged != 1) {
      MFEM_WARNING("CG convection solver did not converge!");
   }
   else {
      int num_procs, myid;
      MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
      MPI_Comm_rank(MPI_COMM_WORLD, &myid);

      if (myid == 0) {
         int iter = gmres.GetNumIterations();
         double residual = gmres.GetFinalNorm();
         std::cout << "CG convection solver converged in " << iter << " iterations.\n";
         std::cout << "Final convection residual norm: " << residual << "\n\n";
      }
   }
   
   delete amg;
   delete a;
   delete b;
   delete A;

   return converged;
   
}

ParGridFunction* CGConvectionSolver::GetSolutionField()
{
   return m_sol;
}

CGConvectionSolver::~CGConvectionSolver()
{
   delete m_fe_coll;
   delete m_fe_space;
   delete m_sol;
}

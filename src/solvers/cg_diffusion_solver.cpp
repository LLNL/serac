#include "cg_diffusion_solver.hpp"

using namespace mfem;

CGDiffusionSolver::CGDiffusionSolver(ParMesh *pmesh, int order) :
   m_pmesh(pmesh), m_order(order), m_source(nullptr), m_diffusion(nullptr), m_neumann_values(nullptr),
   m_dirichlet_values(nullptr)
{
   m_fe_coll = new H1_FECollection(m_order, m_pmesh->Dimension());
   m_fe_space = new ParFiniteElementSpace(m_pmesh, m_fe_coll);
   m_sol = new ParGridFunction(m_fe_space);
}

void CGDiffusionSolver::SetSource(Coefficient & source)
{
   m_source = &source;
}

void CGDiffusionSolver::SetDiffusionCoefficient(Coefficient & diff)
{
   m_diffusion = &diff;
}

void CGDiffusionSolver::SetNeumannBoundary(Array<int> &neumann_indicator, Coefficient & neumann_values)
{
   m_neumann_indicator = neumann_indicator;
   m_neumann_values = &neumann_values;
}

void CGDiffusionSolver::SetDirichletBoundary(Array<int> &dirichlet_indicator, Coefficient & dirichlet_values)
{
   m_dirichlet_indicator = dirichlet_indicator;
   m_dirichlet_values = &dirichlet_values;
}

int CGDiffusionSolver::Solve()
{
   *m_sol = 0.0;   
   Array<int> ess_tdof_list;

   // Create the linear form RHS object   
   ParLinearForm *b = new ParLinearForm(m_fe_space);
   if (m_source != nullptr) {
      // add the volume source if defined
      b->AddDomainIntegrator(new DomainLFIntegrator(*m_source));
   }
   if (m_dirichlet_values != nullptr) {
      // Get the essential dof numbers 
      m_fe_space->GetEssentialTrueDofs(m_dirichlet_indicator, ess_tdof_list);
      // Project the boundary values onto the essential dofs
      m_sol->ProjectBdrCoefficient(*m_dirichlet_values, m_dirichlet_indicator);
   }
   if (m_neumann_values != nullptr) {
      // Add the outflow boundary integrator
      b->AddBoundaryIntegrator(new BoundaryLFIntegrator(*m_neumann_values), m_neumann_indicator);
   }

   // Assemble the linear form object   
   b->Assemble();

   // Create the bilinear form object   
   ParBilinearForm *a = new ParBilinearForm(m_fe_space);
   a->AddDomainIntegrator(new DiffusionIntegrator(*m_diffusion));
   a->Assemble();

   // Create the linear algebra objects for the solve   
   HypreParMatrix *A = new HypreParMatrix;
   Vector B, X;

   // Form the linear algebra system from the abstract objects and apply boundary conditions   
   a->FormLinearSystem(ess_tdof_list, *m_sol, *b, *A, X, B);

   // Setup the solver parameters   
   HypreBoomerAMG *amg = new HypreBoomerAMG(*A);
   amg->SetPrintLevel(0);
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetAbsTol(0.0);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(200);
   cg.SetPrintLevel(0);
   cg.SetOperator(*A);
   cg.SetPreconditioner(*amg);

   // Solve the system and test for convergence   
   cg.Mult(B, X);
   int converged = cg.GetConverged();

   if (converged != 1) {
      MFEM_WARNING("CG diffusion solver did not converge!");
   }
   else {
      int num_procs, myid;
      MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
      MPI_Comm_rank(MPI_COMM_WORLD, &myid);

      if (myid == 0) {
         int iter = cg.GetNumIterations();
         double residual = cg.GetFinalNorm();
         std::cout << "CG diffusion solver converged in " << iter << " iterations.\n";
         std::cout << "Final diffusion residual norm: " << residual << "\n\n";
      }
   }
   
   a->RecoverFEMSolution(X, *b, *m_sol);
   
   delete amg;
   delete a;
   delete b;
   delete A;

   return converged;

}

ParGridFunction* CGDiffusionSolver::GetSolutionField()
{
   return m_sol;
}

CGDiffusionSolver::~CGDiffusionSolver()
{
   delete m_fe_coll;
   delete m_fe_space;
   delete m_sol;
}

#include "dg_diffusion_solver.hpp"
#include "integrators/dg_neumann_integrator.hpp"

using namespace mfem;

DGDiffusionSolver::DGDiffusionSolver(ParMesh *pmesh, int order, double kappa, double sigma) :
   m_pmesh(pmesh), m_order(order), m_source(nullptr), m_diffusion(nullptr), m_neumann_values(nullptr),
   m_dirichlet_values(nullptr), m_sigma(sigma), m_kappa(kappa)
{
   m_fe_coll = new DG_FECollection(m_order, m_pmesh->Dimension());
   m_fe_space = new ParFiniteElementSpace(m_pmesh, m_fe_coll);
   m_sol = new ParGridFunction(m_fe_space);
}

void DGDiffusionSolver::SetSource(Coefficient & source)
{
   m_source = &source;
}

void DGDiffusionSolver::SetDiffusionCoefficient(Coefficient & diff)
{
   m_diffusion = &diff;
}

void DGDiffusionSolver::SetNeumannBoundary(Array<int> &neumann_indicator, Coefficient & neumann_values)
{
   m_neumann_indicator = neumann_indicator;
   m_neumann_values = &neumann_values;
}

void DGDiffusionSolver::SetDirichletBoundary(Array<int> &dirichlet_indicator, Coefficient & dirichlet_values)
{
   m_dirichlet_indicator = dirichlet_indicator;
   m_dirichlet_values = &dirichlet_values;
}

int DGDiffusionSolver::Solve()
{
   *m_sol = 0.0;

   // Create the linear form RHS object
   ParLinearForm *b = new ParLinearForm(m_fe_space);
   if (m_source != nullptr) {
      // add the volume source if defined
      b->AddDomainIntegrator(new DomainLFIntegrator(*m_source));
   }
   if (m_dirichlet_values != nullptr) {
      // Add the DG boundary integrators for the dirichlet conditions
      b->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(*m_dirichlet_values, *m_diffusion, m_sigma, m_kappa), m_dirichlet_indicator);   
   }
   if (m_neumann_values != nullptr) {
      // Add the DG boundary integrators for the neumann conditions      
      b->AddBdrFaceIntegrator(new DGNeumannLFIntegrator(*m_neumann_values), m_neumann_indicator);
   }

   // Assemble the linear form object   
   b->Assemble();

   // Create the bilinear form object      
   ParBilinearForm *a = new ParBilinearForm(m_fe_space);

   // Add the DG diffusion integrators
   a->AddDomainIntegrator(new DiffusionIntegrator(*m_diffusion));
   a->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(*m_diffusion, m_sigma, m_kappa));
   a->AddBdrFaceIntegrator(new DGDiffusionIntegrator(*m_diffusion, m_sigma, m_kappa));

   // Assemble the bilinear form object
   a->Assemble();
   a->Finalize();

   // Create the linear algebra objects for the solve   
   HypreParMatrix *A = a->ParallelAssemble();
   HypreParVector *B = b->ParallelAssemble();
   HypreParVector *X = m_sol->ParallelProject();

   // Setup solver parameters
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

   // Solve the system   
   gmres.Mult(*B, *X);
   int converged = gmres.GetConverged();

   // Communicate the shared nodes
   *m_sol = *X;
   
   if (converged != 1) {
      MFEM_WARNING("DG diffusion solver did not converge!");
   }
   else {
      int num_procs, myid;
      MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
      MPI_Comm_rank(MPI_COMM_WORLD, &myid);

      if (myid == 0) {
         int iter = gmres.GetNumIterations();
         double residual = gmres.GetFinalNorm();
         std::cout << "DG diffusion solver converged in " << iter << " iterations.\n";
         std::cout << "Final diffusion residual norm: " << residual << "\n\n";
      }
   }
   
   delete amg;
   delete a;
   delete b;
   delete A;
   delete B;
   delete X;

   return converged;

}

ParGridFunction* DGDiffusionSolver::GetSolutionField()
{
   return m_sol;
}

DGDiffusionSolver::~DGDiffusionSolver()
{
   delete m_fe_coll;
   delete m_fe_space;
   delete m_sol;
}

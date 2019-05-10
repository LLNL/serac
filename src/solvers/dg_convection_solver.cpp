#include "dg_convection_solver.hpp"
#include "integrators/dg_neumann_integrator.hpp"

using namespace mfem;

DGConvectionSolver::DGConvectionSolver(ParMesh *pmesh, int order, double kappa, double sigma) :
   m_pmesh(pmesh), m_order(order), m_source(nullptr), m_diffusion(nullptr), m_convection(nullptr), m_outflow_values(nullptr),
   m_inflow_values(nullptr), m_sigma(sigma), m_kappa(kappa)
{
   m_fe_coll = new DG_FECollection(m_order, m_pmesh->Dimension());
   m_fe_space = new ParFiniteElementSpace(m_pmesh, m_fe_coll);
   m_sol = new ParGridFunction(m_fe_space);
}

void DGConvectionSolver::SetSource(Coefficient & source)
{
   m_source = &source;
}

void DGConvectionSolver::SetDiffusionCoefficient(Coefficient & diff)
{
   m_diffusion = &diff;
}

void DGConvectionSolver::SetConvectionCoefficient(VectorCoefficient & conv)
{
   m_convection = &conv;
}

void DGConvectionSolver::SetOutflowBoundary(Array<int> &outflow_indicator, Coefficient & outflow_values)
{
   m_outflow_indicator = outflow_indicator;
   m_outflow_values = &outflow_values;
}

void DGConvectionSolver::SetInflowBoundary(Array<int> &inflow_indicator, Coefficient & inflow_values)
{
   m_inflow_indicator = inflow_indicator;
   m_inflow_values = &inflow_values;
}

int DGConvectionSolver::Solve()
{

   *m_sol = 0.0;

   // Create the linear form RHS object   
   ParLinearForm *b = new ParLinearForm(m_fe_space);
   if (m_source != nullptr) {
      // add the volume source if defined
      b->AddDomainIntegrator(new DomainLFIntegrator(*m_source));
   }

   if (m_inflow_values != nullptr) {
      // Add the DG boundary integrators for the inflow conditions
      b->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(*m_inflow_values, *m_diffusion, m_sigma, m_kappa), m_inflow_indicator);   
      b->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(*m_inflow_values, *m_convection, -1.0, -0.5), m_inflow_indicator);
   }

   if (m_outflow_values != nullptr) {
      // Add the DG boundary integrators for the outflow conditions
      b->AddBdrFaceIntegrator(new DGNeumannLFIntegrator(*m_outflow_values), m_outflow_indicator);
   }

   // Assemble the linear form object
   b->Assemble();

   // Create the bilinear form object   
   ParBilinearForm *a = new ParBilinearForm(m_fe_space);

   // Add the DG diffusion and convection integrators
   a->AddDomainIntegrator(new DiffusionIntegrator(*m_diffusion));
   a->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(*m_diffusion, m_sigma, m_kappa));
   a->AddBdrFaceIntegrator(new DGDiffusionIntegrator(*m_diffusion, m_sigma, m_kappa));

   a->AddDomainIntegrator(new ConvectionIntegrator(*m_convection));
   a->AddInteriorFaceIntegrator(new TransposeIntegrator(new DGTraceIntegrator(*m_convection, -1.0, 0.5)));
   a->AddBdrFaceIntegrator(new TransposeIntegrator(new DGTraceIntegrator(*m_convection, -1.0, 0.5)));   

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
      MFEM_WARNING("DG convection solver did not converge!");
   }
   else {
      int num_procs, myid;
      MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
      MPI_Comm_rank(MPI_COMM_WORLD, &myid);

      if (myid == 0) {
         int iter = gmres.GetNumIterations();
         double residual = gmres.GetFinalNorm();
         std::cout << "DG convection solver converged in " << iter << " iterations.\n";
         std::cout << "Final convection residual norm: " << residual << "\n\n";
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

ParGridFunction* DGConvectionSolver::GetSolutionField()
{
   return m_sol;
}

DGConvectionSolver::~DGConvectionSolver()
{
   delete m_fe_coll;
   delete m_fe_space;
   delete m_sol;
}

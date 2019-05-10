#include "potential_solver.hpp"
#include "coefficients/flobat_scalar_coefficient.hpp"

using namespace mfem;

PotentialSolver::PotentialSolver(ParMesh *pmesh, int order, FlobatParams & params) :
   m_params(params)
{
   m_phi1_solver = new CGDiffusionSolver(pmesh, order);
   m_phi2_solver = new CGDiffusionSolver(pmesh, order);   
}

void PotentialSolver::SetGamma(Coefficient & gamma)
{
   m_gamma = &gamma;
}

void PotentialSolver::SetPhi1(Coefficient & phi1)
{
   m_phi1_guess = &phi1;
}

void PotentialSolver::SetPhi2(Coefficient & phi2)
{
   m_phi2_guess = &phi2;
}

void PotentialSolver::SetCO(Coefficient & CO)
{
   m_CO = &CO;
}

void PotentialSolver::SetCR(Coefficient & CR)
{
   m_CR = &CR;
}

void PotentialSolver::SetVelocity(VectorCoefficient & vel)
{
   m_vel = &vel;
}

void PotentialSolver::SetCollectorBoundary(Array<int> &collector_indicator)
{
   m_collector_indicator = collector_indicator;
}

void PotentialSolver::SetMembraneBoundary(Array<int> &membrane_indicator)
{
   m_membrane_indicator = membrane_indicator;
}

int PotentialSolver::Solve()
{
   // Generate appropriate coefficients
   ConstantCoefficient negone(-1.0);
   ConstantCoefficient zero(0.0);
   
   PotentialSourceCoefficient source(*m_gamma, *m_phi1_guess, *m_phi2_guess, *m_CO, *m_CR, *m_vel, m_params);
   ProductCoefficient negsource(source, negone);
                                
   SigmaCoefficient sigma(*m_gamma, *m_phi1_guess, *m_phi2_guess, *m_CO, *m_CR, *m_vel, m_params);
   KappaCoefficient kappa(*m_gamma, *m_phi1_guess, *m_phi2_guess, *m_CO, *m_CR, *m_vel, m_params);      

   double I = m_params.GetValue(FlobatParams::I);
   double A = m_params.GetValue(FlobatParams::A);   
   ConstantCoefficient flux(I/A);

   // Setup the generic CG diffusion solvers
   m_phi1_solver->SetSource(source);
   m_phi1_solver->SetDiffusionCoefficient(sigma);
   m_phi1_solver->SetDirichletBoundary(m_collector_indicator, zero);

   m_phi2_solver->SetSource(negsource);
   m_phi2_solver->SetDiffusionCoefficient(kappa);
   m_phi2_solver->SetNeumannBoundary(m_membrane_indicator, flux);

   // Solve the systems
   int phi1_converged = m_phi1_solver->Solve();
   if (phi1_converged != 1) {
      MFEM_WARNING("Phi1 solver did not converge!");      
   }
      
   int phi2_converged = m_phi2_solver->Solve();   
   if (phi2_converged != 1) {
      MFEM_WARNING("Phi2 solver did not converge!");      
   }

   m_phi1_result = m_phi1_solver->GetSolutionField();
   m_phi2_result = m_phi2_solver->GetSolutionField();   

   if ((phi1_converged != 1) || (phi2_converged != 1)) {
      return 0;
   }
      
   return 1;
}

ParGridFunction* PotentialSolver::GetPhi1Field()
{
   return m_phi1_result;
}

ParGridFunction* PotentialSolver::GetPhi2Field()
{
   return m_phi2_result;
}


PotentialSolver::~PotentialSolver()
{
   delete m_phi1_solver;
   delete m_phi2_solver;
}

#ifndef POTENTIAL_SOLVER
#define POTENTIAL_SOLVER

#include "mfem.hpp"
#include "models/parameters.hpp"
#include "solvers/cg_diffusion_solver.hpp"

using namespace mfem;

class PotentialSolver
{
public:
      
   PotentialSolver(ParMesh *pmesh, int order, FlobatParams & params);

   ~PotentialSolver();

   void SetGamma(Coefficient & gamma);
   void SetPhi1(Coefficient & phi1);
   void SetPhi2(Coefficient & phi2);
   void SetCO(Coefficient & CO);
   void SetCR(Coefficient & CR);
   void SetVelocity(VectorCoefficient & vel);      
   
   void SetCollectorBoundary(Array<int> & collector_indicator);
   void SetMembraneBoundary(Array<int> & membrane_indicator);

   int Solve();
   
   ParGridFunction* GetPhi1Field();
   ParGridFunction* GetPhi2Field();   
   
private:
   
   Array<int> m_collector_indicator;
   Array<int> m_membrane_indicator;

   Coefficient *m_gamma;
   Coefficient *m_phi1_guess;
   Coefficient *m_phi2_guess;   
   Coefficient *m_CR;
   Coefficient *m_CO;
   VectorCoefficient *m_vel;
   
   FlobatParams m_params;

   CGDiffusionSolver* m_phi1_solver;
   CGDiffusionSolver* m_phi2_solver;   
   
   ParGridFunction *m_phi1_result;
   ParGridFunction *m_phi2_result;   
   
};

#endif

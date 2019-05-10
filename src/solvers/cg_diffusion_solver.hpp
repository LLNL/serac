#ifndef CG_DIFFUSION_SOLVER
#define CG_DIFFUSION_SOLVER

#include "mfem.hpp"

using namespace mfem;

class CGDiffusionSolver
{
public:
   CGDiffusionSolver(ParMesh *pmesh, int order);

   ~CGDiffusionSolver();
   
   void SetSource(Coefficient & source);

   void SetDiffusionCoefficient(Coefficient & diff);

   void SetNeumannBoundary(Array<int> & neumann_indicator, Coefficient & neumann_values);

   void SetDirichletBoundary(Array<int> & dirichlet_indicator, Coefficient & dirichlet_values);
   
   int Solve();
   
   ParGridFunction* GetSolutionField();
   
private:

   ParMesh *m_pmesh;
   int m_order;
   Coefficient *m_source;
   Coefficient *m_diffusion;
   Array<int> m_neumann_indicator;
   Array<int> m_dirichlet_indicator;
   Coefficient *m_neumann_values;
   Coefficient *m_dirichlet_values;

   FiniteElementCollection *m_fe_coll;
   ParFiniteElementSpace *m_fe_space;
   
   ParGridFunction *m_sol;
   
};

#endif

#ifndef CG_CONVECTION_SOLVER
#define CG_CONVECTION_SOLVER

#include "mfem.hpp"

using namespace mfem;

class CGConvectionSolver
{
public:
   CGConvectionSolver(ParMesh *pmesh, int order);

   ~CGConvectionSolver();
   
   void SetSource(Coefficient & source);

   void SetDiffusionCoefficient(Coefficient & diff);

   void SetConvectionCoefficient(VectorCoefficient & conv);   

   void SetOutflowBoundary(Array<int> & outflow_indicator, Coefficient & outflow_values);

   void SetInflowBoundary(Array<int> & inflow_indicator, Coefficient & inflow_values);

   int Solve();
      
   ParGridFunction* GetSolutionField();
   
private:

   ParMesh *m_pmesh;
   int m_order;
   Coefficient *m_source;
   Coefficient *m_diffusion;
   VectorCoefficient *m_convection;   
   Array<int> m_outflow_indicator;
   Array<int> m_inflow_indicator;
   Coefficient *m_outflow_values;
   Coefficient *m_inflow_values;

   FiniteElementCollection *m_fe_coll;
   ParFiniteElementSpace *m_fe_space;
   
   ParGridFunction *m_sol;
   
};

#endif

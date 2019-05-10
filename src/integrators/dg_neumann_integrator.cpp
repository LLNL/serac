#include "dg_neumann_integrator.hpp"

using namespace mfem;

void DGNeumannLFIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                                   ElementTransformation &Tr,
                                                   Vector &elvect)
{
   mfem_error("DGNeumannLFIntegrator::AssembleRHSElementVect(...)");

   // Quiet the compiler
   elvect = elvect;
   int dim = Tr.GetDimension();
   dim = el.GetDim();
}
   

void DGNeumannLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   int dof  = el.GetDof();
   
   shape.SetSize(dof);
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL) {
      int intorder = el.GetOrder() + 1;
      ir = &IntRules.Get(Tr.FaceGeom, intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++) {
      const IntegrationPoint &ip = ir->IntPoint(i);
      IntegrationPoint eip;
      Tr.Loc1.Transform(ip, eip);

      el.CalcShape(eip, shape);
      
      Tr.Face->SetIntPoint(&ip);
      double val = Q->Eval(*Tr.Face, ip);
      val *= Tr.Face->Weight() * ip.weight;
      for (int s = 0; s < dof; s++) {
         elvect(s) += val * shape(s);
      }
   }
}

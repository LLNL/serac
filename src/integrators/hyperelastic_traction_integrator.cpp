#include "hyperelastic_traction_integrator.hpp"

using namespace mfem;

void HyperelasticTractionIntegrator::AssembleFaceVector(const FiniteElement &el1,
                                                        __attribute__((unused)) const FiniteElement &el2,
                                                        FaceElementTransformations &Tr,
                                                        const Vector &elfun, 
                                                        Vector &elvec)
{

   int dim = el1.GetDim();
   int dof = el1.GetDof();

   shape.SetSize (dof);
   elvec.SetSize (dim*dof);

   DSh_u.SetSize(dof, dim);
   DS_u.SetSize(dof, dim);
   J0i.SetSize(dim);
   F.SetSize(dim);
   Finv.SetSize(dim);

   PMatI_u.UseExternalData(elfun.GetData(), dof, dim);

   int intorder = 2*el1.GetOrder() + 3; 
   const IntegrationRule &ir = IntRules.Get(Tr.FaceGeom, intorder);

   elvec = 0.0;

   Vector trac(dim);
   Vector ftrac(dim);
   Vector nor(dim);
   Vector fnor(dim);
   Vector u(dim);
   Vector fu(dim);
   
   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      IntegrationPoint eip;
      Tr.Loc1.Transform(ip, eip);
      
      Tr.Face->SetIntPoint(&ip);

      CalcOrtho(Tr.Face->Jacobian(), nor);

      //Normalize vector
      double norm = nor.Norml2();
      nor /= norm;
      
      //Compute traction
      function.Eval(trac, *Tr.Face, ip);
      
      Tr.Elem1->SetIntPoint(&eip);
      CalcInverse(Tr.Elem1->Jacobian(), J0i);
      
      el1.CalcDShape(eip, DSh_u);
      Mult(DSh_u, J0i, DS_u);
      MultAtB(PMatI_u, DS_u, F);

      CalcInverse(F, Finv);

      Finv.MultTranspose(nor, fnor);

      el1.CalcShape (eip, shape);
      for (int j = 0; j < dof; j++) {
         for (int k = 0; k < dim; k++) {
            u = 0.0;
            u(k) = shape(j);
            F.Mult(u, fu);
            F.Mult(trac, ftrac);
            for (int l=0; l < dim; l++) {
               elvec(dof*k+j) += ftrac(l) * fu(l) * ip.weight * Tr.Face->Weight() * F.Det() * fnor.Norml2();
            }
         }
      }
   }
}

void HyperelasticTractionIntegrator::AssembleFaceGrad(const FiniteElement &el1,
                                                      __attribute__((unused)) const FiniteElement &el2,
                                                     FaceElementTransformations &Tr,
                                                     const Vector &elfun, 
                                                     DenseMatrix &elmat)
{
   int dof = el1.GetDof();
   int dim = el1.GetDim();

   double small = 1.0e-5;
   
   Vector test1(dim*dof);
   Vector test2(dim*dof);   
   Vector result1(dim*dof);
   Vector result2(dim*dof);
   
   elmat.SetSize(dof*dim, dof*dim);
   
   for (int i = 0; i<dof; i++) {
      for (int i_dim = 0; i_dim < dim; i_dim++) {
         
         test1 = elfun;
         test1(dof*i_dim + i) += small;
         AssembleFaceVector(el1, el2, Tr, test1, result1);

         test2 = elfun;
         test2(dof*i_dim + i) -= small;
         AssembleFaceVector(el1, el2, Tr, test2, result2);

         for (int j = 0; j<dof; j++) {
            for (int j_dim = 0; j_dim < dim; j_dim++) {
               elmat(dof*j_dim + j, dof*i_dim + i) = (test1(dof*j_dim + j) - test2(dof*j_dim + j))/(2.0 * small);
            }
         }            
      }
   }  
}

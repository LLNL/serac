#include "hyperelastic_traction_integrator.hpp"

using namespace mfem;

void HyperelasticTractionIntegrator::AssembleFaceVector(const FiniteElement &el1,
                                                        const FiniteElement &el2,
                                                        FaceElementTransformations &Tr,
                                                        const Vector &elfun, 
                                                        Vector &elvec)
{

   int dim = el1.GetDim();
   int dof_u = el1.GetDof();

   shape.SetSize (dof_u);
   elvec.SetSize (dim*dof_u);

   DSh_u.SetSize(dof_u, dim);
   DS_u.SetSize(dof_u, dim);
   J0i.SetSize(dim);
   J.SetSize(dim);
   Jinv.SetSize(dim);

   PMatI_u.UseExternalData(elfun.GetData(), dof_u, dim);

   int intorder = 2*el1.GetOrder() + 3; 
   const IntegrationRule &ir = IntRules.Get(Tr.FaceGeom, intorder);

   elvec = 0.0;

   Vector trac(dim);
   Vector ftrac(dim);
   
   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      IntegrationPoint eip;
      Tr.Loc1.Transform(ip, eip);
      
      Tr.Face->SetIntPoint(&ip);

      //Compute traction
      function.Eval(trac, *Tr.Face, ip);

      Tr.Elem1->SetIntPoint(&eip);
      CalcInverse(Tr.Elem1->Jacobian(), J0i);
      
      el1.CalcDShape(eip, DSh_u);
      Mult(DSh_u, J0i, DS_u);
      MultAtB(PMatI_u, DS_u, J);

      CalcInverse(J, Jinv);

      Jinv.MultTranspose(trac, ftrac);

      el1.CalcShape (eip, shape);
      for (int j = 0; j < dof_u; j++)
         for (int k = 0; k < dim; k++)
         {
            elvec(dof_u*k+j) += ip.weight * Tr.Face->Weight() * ftrac(k) * shape(j) * J.Det();
         }
   }
}

void HyperelasticTractionIntegrator::AssembleFaceGrad(const FiniteElement &el1,
                                                     const FiniteElement &el2,
                                                     FaceElementTransformations &Tr,
                                                     const Vector &elfun, 
                                                     DenseMatrix &elmat)
{
   int dof_u = el1.GetDof();

   int dim = el1.GetDim();

   elmat.SetSize(dof_u*dim, dof_u*dim);

   elmat = 0.0;

   shape.SetSize (dof_u);
   nor.SetSize (dim);
   fnor.SetSize (dim);
   
   DSh_u.SetSize(dof_u, dim);
   DS_u.SetSize(dof_u, dim);
   Sh_u.SetSize(dof_u);
   J0i.SetSize(dim);
   J.SetSize(dim);
   Jinv.SetSize(dim);
   JinvT.SetSize(dim);

   PMatI_u.UseExternalData(elfun.GetData(), dof_u, dim);

   int intorder = 2*el1.GetOrder() + 3; 
   const IntegrationRule &ir = IntRules.Get(Tr.FaceGeom, intorder);

   double dJ;
   Vector trac(dim);
   Vector ftrac(dim);

   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      IntegrationPoint eip;
      Tr.Loc1.Transform(ip, eip);

      Tr.Face->SetIntPoint(&ip);

      Tr.Elem1->SetIntPoint(&eip);
      CalcInverse(Tr.Elem1->Jacobian(), J0i);

      el1.CalcDShape(eip, DSh_u);
      el1.CalcShape(eip, Sh_u);
      Mult(DSh_u, J0i, DS_u);
      MultAtB(PMatI_u, DS_u, J);

      CalcInverse(J, Jinv);
      CalcInverseTranspose(J, JinvT);

      dJ = J.Det();

      function.Eval(trac, *Tr.Face, ip);
            
      // u,u block
      for (int i_u = 0; i_u < dof_u; i_u++) {
      for (int i_dim = 0; i_dim < dim; i_dim++) {
         for (int j_u = 0; j_u < dof_u; j_u++) {
         for (int j_dim = 0; j_dim < dim; j_dim++) {

            for (int n=0; n<dim; n++) {
            for (int l=0; l<dim; l++) {
               elmat(i_u + i_dim*dof_u, j_u + j_dim*dof_u) += dJ * JinvT(i_dim,l) * JinvT(j_dim,n) * trac(n) * Sh_u(i_u) * DS_u(j_u,n) * ip.weight * Tr.Face->Weight();        
               elmat(i_u + i_dim*dof_u, j_u + j_dim*dof_u) -= dJ * JinvT(i_dim,n) * JinvT(j_dim,l) * trac(n) * Sh_u(i_u) * DS_u(j_u,n) * ip.weight * Tr.Face->Weight();
               
            }
            }
            
            
         }
         }
      }
      }
      
   }
}

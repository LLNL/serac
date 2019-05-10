#include "mfem.hpp"

using namespace mfem;

class DGNeumannLFIntegrator : public LinearFormIntegrator
{
protected:
   Coefficient *Q;
   Vector shape;

public:
   DGNeumannLFIntegrator(Coefficient &u)
      : Q(&u) {}

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);
};

// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause) 

#include "mfem.hpp"

using namespace mfem;

class HyperelasticTractionIntegrator : public NonlinearFormIntegrator
{
private:
   VectorCoefficient &function;
   mutable DenseMatrix DSh_u, DS_u, J0i, F, Finv, FinvT, PMatI_u;
   mutable Vector shape, nor, fnor, Sh_p, Sh_u;
   
public:
   HyperelasticTractionIntegrator(VectorCoefficient &f) : function(f) { }

   virtual void AssembleFaceVector(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Tr,
                                   const Vector &elfun, 
                                   Vector &elvec);

   virtual void AssembleFaceGrad(const FiniteElement &el1,
                                 const FiniteElement &el2,
                                 FaceElementTransformations &Tr,
                                 const Vector &elfun, 
                                 DenseMatrix &elmat);

   virtual ~HyperelasticTractionIntegrator() { }
};

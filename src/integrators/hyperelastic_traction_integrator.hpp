// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "mfem.hpp"

class HyperelasticTractionIntegrator : public mfem::NonlinearFormIntegrator
{
private:
  mfem::VectorCoefficient &function;
  mutable mfem::DenseMatrix DSh_u, DS_u, J0i, F, Finv, FinvT, PMatI_u;
  mutable mfem::Vector shape, nor, fnor, Sh_p, Sh_u;

public:
  HyperelasticTractionIntegrator(mfem::VectorCoefficient &f) : function(f) { }

  virtual void AssembleFaceVector(const mfem::FiniteElement &el1,
                                  const mfem::FiniteElement &el2,
                                  mfem::FaceElementTransformations &Tr,
                                  const mfem::Vector &elfun,
                                  mfem::Vector &elvec);

  virtual void AssembleFaceGrad(const mfem::FiniteElement &el1,
                                const mfem::FiniteElement &el2,
                                mfem::FaceElementTransformations &Tr,
                                const mfem::Vector &elfun,
                                mfem::DenseMatrix &elmat);

  virtual ~HyperelasticTractionIntegrator() { }
};

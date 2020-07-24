// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "mfem.hpp"

class HyperelasticTractionIntegrator : public mfem::NonlinearFormIntegrator {
 private:
  mfem::VectorCoefficient & function_;
  mutable mfem::DenseMatrix DSh_u_, DS_u_, J0i_, F_, Finv_, FinvT_, PMatI_u_;
  mutable mfem::Vector      shape_, nor_, fnor_, Sh_p_, Sh_u_;

 public:
  HyperelasticTractionIntegrator(mfem::VectorCoefficient &f) : function_(f) {}

  virtual void AssembleFaceVector(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2,
                                  mfem::FaceElementTransformations &Tr, const mfem::Vector &elfun, mfem::Vector &elvec);

  virtual void AssembleFaceGrad(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2,
                                mfem::FaceElementTransformations &Tr, const mfem::Vector &elfun,
                                mfem::DenseMatrix &elmat);

  virtual ~HyperelasticTractionIntegrator() {}
};

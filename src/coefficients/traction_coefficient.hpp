// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef TRACTION_COEF
#define TRACTION_COEF

#include "mfem.hpp"

class VectorScaledConstantCoefficient : public mfem::VectorCoefficient {
 private:
  mfem::Vector vec_;
  double       scale_;

 public:
  VectorScaledConstantCoefficient(const mfem::Vector &v) : mfem::VectorCoefficient(v.Size()), vec_(v) {}
  using mfem::VectorCoefficient::Eval;
  void         SetScale(double s) { scale_ = s; }
  virtual void Eval(mfem::Vector &V, mfem::ElementTransformation &, const mfem::IntegrationPoint &)
  {
    V = vec_;
    V *= scale_;
  }
};

#endif

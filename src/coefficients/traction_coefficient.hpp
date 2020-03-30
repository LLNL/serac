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
  mfem::Vector vec;
  double       scale;

 public:
  VectorScaledConstantCoefficient(const mfem::Vector &v) : mfem::VectorCoefficient(v.Size()), vec(v) {}
  using mfem::VectorCoefficient::Eval;
  void         SetScale(double s) { scale = s; }
  virtual void Eval(mfem::Vector &V, __attribute__((unused)) mfem::ElementTransformation &T,
                    __attribute__((unused)) const mfem::IntegrationPoint &ip)
  {
    V = vec;
    V *= scale;
  }
};

#endif

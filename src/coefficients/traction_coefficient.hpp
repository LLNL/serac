// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause) 

#ifndef TRACTION_COEF
#define TRACTION_COEF

#include "mfem.hpp"

using namespace mfem;


class VectorScaledConstantCoefficient : public VectorCoefficient
{
private:
   Vector vec;
   double scale;
public:
   VectorScaledConstantCoefficient(const Vector &v)
      : VectorCoefficient(v.Size()), vec(v) { }
   using VectorCoefficient::Eval;
   void SetScale(double s) { scale = s; }
   virtual void Eval(Vector &V, __attribute__((unused)) ElementTransformation &T,
                     __attribute__((unused)) const IntegrationPoint &ip) { V = vec; V *= scale; }
};

#endif

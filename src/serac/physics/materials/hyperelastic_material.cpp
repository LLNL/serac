// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/materials/hyperelastic_material.hpp"

namespace serac {

inline void NeoHookeanMaterial::EvalCoeffs() const
{
  mu = c_mu->Eval(*Ttr, Ttr->GetIntPoint());
  K  = c_K->Eval(*Ttr, Ttr->GetIntPoint());
  if (c_g) {
    g = c_g->Eval(*Ttr, Ttr->GetIntPoint());
  }
}

double NeoHookeanMaterial::EvalW(const mfem::DenseMatrix& F) const
{
  int dim = F.Width();

  if (have_coeffs) {
    EvalCoeffs();
  }

  double detF = F.Det();
  double sJ   = detF / g;
  double bI1  = pow(detF, -2.0 / dim) * (F * F);  // \bar{I}_1

  return 0.5 * (mu * (bI1 - dim) + K * (sJ - 1.0) * (sJ - 1.0));
}

void NeoHookeanMaterial::EvalP(const mfem::DenseMatrix& F, mfem::DenseMatrix& P) const
{
  int dim = F.Width();

  if (have_coeffs) {
    EvalCoeffs();
  }

  FinvT.SetSize(dim);
  CalcInverseTranspose(F, FinvT);
  double detF = F.Det();

  double a = mu * pow(detF, -2.0 / dim);
  double b = K * detF * (detF / g - 1.0) / g - a * (F * F) / (dim);

  P = 0.0;
  P.Add(a, F);
  P.Add(b, FinvT);
}

void NeoHookeanMaterial::AssembleTangentModuli(const mfem::DenseMatrix& F, const mfem::DenseMatrix& B0_T,
                                               const double weight, mfem::DenseMatrix& T) const
{
  int dof = B0_T.Height(), dim = B0_T.Width();

  if (have_coeffs) {
    EvalCoeffs();
  }

  FinvT.SetSize(dim);
  G.SetSize(dof, dim);
  C.SetSize(dof, dim);

  double detF = F.Det();
  double sJ   = detF / g;
  double a    = mu * pow(detF, -2.0 / dim);
  double bc   = a * (F * F) / dim;
  double b    = bc - K * sJ * (sJ - 1.0);
  double c    = 2.0 * bc / dim + K * sJ * (2.0 * sJ - 1.0);

  CalcInverseTranspose(F, FinvT);

  MultABt(B0_T, F, C);      // C = B0_T F^t
  MultABt(B0_T, FinvT, G);  // G = B0_T F^{-1}

  a *= weight;
  b *= weight;
  c *= weight;

  // 1.
  for (int i = 0; i < dof; i++)
    for (int k = 0; k <= i; k++) {
      double s = 0.0;
      for (int d = 0; d < dim; d++) {
        s += B0_T(i, d) * B0_T(k, d);
      }
      s *= a;

      for (int d = 0; d < dim; d++) {
        T(i + d * dof, k + d * dof) += s;
      }

      if (k != i)
        for (int d = 0; d < dim; d++) {
          T(k + d * dof, i + d * dof) += s;
        }
    }

  a *= (-2.0 / dim);

  // 2.
  for (int i = 0; i < dof; i++)
    for (int j = 0; j < dim; j++)
      for (int k = 0; k < dof; k++)
        for (int l = 0; l < dim; l++) {
          T(i + j * dof, k + l * dof) +=
              a * (C(i, j) * G(k, l) + G(i, j) * C(k, l)) + b * G(i, l) * G(k, j) + c * G(i, j) * G(k, l);
        }
}

}  // namespace serac
// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/materials/hyperelastic_material.hpp"

#include "serac/infrastructure/logger.hpp"

#include <cmath>

namespace serac {

inline void NeoHookeanMaterial::EvalCoeffs() const
{
  mu_     = c_mu_->Eval(*Ttr_, Ttr_->GetIntPoint());
  bulk_ = c_bulk_->Eval(*Ttr_, Ttr_->GetIntPoint());
}

double NeoHookeanMaterial::EvalW(const mfem::DenseMatrix& F) const
{

  int dim = F.Width();

  SLIC_ERROR_IF(dim != 2 && dim != 3, "NeoHookean material used for spatial dimension not 2 or 3!");

  if (c_mu_) {
    EvalCoeffs();
  }

  double J = F.Det();
  double I1_bar = pow(J, -2.0/dim)*(F*F); // \bar{I}_1

  return 0.5*(mu_*(I1_bar - dim) + bulk_*(J - 1.0)*(J - 1.0));
}

void NeoHookeanMaterial::EvalStress(const mfem::DenseMatrix& F, mfem::DenseMatrix& sigma) const
{
  int dim = F.Width();
  B_.SetSize(dim);
  sigma.SetSize(dim);

  if (c_mu_)
  {
    EvalCoeffs();
  }

  double dJ = F.Det();

  double a  = mu_*pow(dJ, -2.0/dim);
  double b  = bulk_*(dJ - 1.0) - a*(F*F)/(dim*dJ);

  mfem::MultABt(F, F, B_);

  sigma = 0.0;

  sigma.Add(a/dJ, B_);

  for (int i=0; i < dim; ++i) {
    sigma(i,i) += b;
  }
}

void NeoHookeanMaterial::AssembleTangentModuli(const mfem::DenseMatrix&, mfem::DenseMatrix&) const
{
  fmt::print("test\n");
}

}  // namespace serac
// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
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
  lambda_ = c_lambda_->Eval(*Ttr_, Ttr_->GetIntPoint());
}

double NeoHookeanMaterial::EvalW(const mfem::DenseMatrix& C) const
{
  int dim = C.Width();

  SLIC_ERROR_IF(dim != 2 && dim != 3, "NeoHookean material used for spatial dimension not 2 or 3!");

  if (c_mu_) {
    EvalCoeffs();
  }

  double J = std::sqrt(C.Det());

  return 0.5 * lambda_ * std::log(J) * std::log(J) - mu_ * std::log(J) + 0.5 * mu_ * (C.Trace() - dim);
}

void NeoHookeanMaterial::EvalPK2(const mfem::DenseMatrix& C, mfem::Vector& S) const
{
  int dim = C.Width();

  SLIC_ERROR_IF(dim != 2 && dim != 3, "NeoHookean material used for spatial dimension not 2 or 3!");

  if (c_mu_) {
    EvalCoeffs();
  }

  double J = std::sqrt(C.Det());

  S.SetSize(dim + shear_terms_.size());
  S_.SetSize(dim);
  Cinv_.SetSize(dim);

  eye_.SetSize(dim);
  eye_ = 0.0;
  for (int i = 0; i < dim; ++i) {
    eye_(i, i) = 1.0;
  }

  CalcInverse(C, Cinv_);

  S_ = 0.0;

  S_.Add(lambda_ * std::log(J), Cinv_);
  S_.Add(mu_, eye_);
  S_.Add(-1.0 * mu_, Cinv_);

  getVoigtVectorFromTensor(shear_terms_, S_, S);
}

void NeoHookeanMaterial::AssembleTangentModuli(const mfem::DenseMatrix& C, mfem::DenseMatrix& T) const
{
  int dim = C.Width();

  SLIC_ERROR_IF(dim != 2 && dim != 3, "NeoHookean material used for spatial dimension not 2 or 3!");

  if (c_mu_) {
    EvalCoeffs();
  }

  T.SetSize(dim + shear_terms_.size());

  CalcInverse(C, Cinv_);
  double J = std::sqrt(C.Det());

  double lambda = lambda_;
  double mu     = mu_ - lambda * std::log(J);

  auto neo_hookean_stiffness = [=](int i, int j, int k, int l) {
    return lambda * Cinv_(i, j) * Cinv_(k, l) + mu * (Cinv_(i, k) * Cinv_(j, l) + Cinv_(i, l) * Cinv_(k, j));
  };

  // Add the volumetric-volumetric terms
  for (int i = 0; i < dim; ++i) {
    for (int j = i; j < dim; ++j) {
      T(i, j) = neo_hookean_stiffness(i, i, j, j);
    }
  }

  for (int j = 0; j < (int)shear_terms_.size(); ++j) {
    for (int i = 0; i < dim; ++i) {
      // Add the volumetric-shear terms
      T(i, dim + j) = neo_hookean_stiffness(i, i, shear_terms_[j].first, shear_terms_[j].second);
    }
    for (int i = 0; i < j; ++i) {
      // Add the shear-shear terms
      T(dim + i, dim + j) = neo_hookean_stiffness(shear_terms_[i].first, shear_terms_[i].second, shear_terms_[j].first,
                                                  shear_terms_[j].second);
    }
  }

  for (int i = 0; i < dim + (int)shear_terms_.size(); ++i) {
    for (int j = 0; j < i; ++j) {
      T(i, j) = T(j, i);
    }
  }
}

}  // namespace serac
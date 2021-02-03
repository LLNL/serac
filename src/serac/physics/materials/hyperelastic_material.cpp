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
  mu_   = c_mu_->Eval(*Ttr_, Ttr_->GetIntPoint());
  bulk_ = c_bulk_->Eval(*Ttr_, Ttr_->GetIntPoint());
}

double NeoHookeanMaterial::EvalStrainEnergy(const mfem::DenseMatrix& F) const
{
  int dim = F.Width();

  SLIC_ERROR_IF(dim != 2 && dim != 3, "NeoHookean material used for spatial dimension not 2 or 3!");

  if (c_mu_) {
    EvalCoeffs();
  }

  double J      = F.Det();
  double I1_bar = pow(J, -2.0 / dim) * (F * F);  // \bar{I}_1

  return 0.5 * (mu_ * (I1_bar - dim) + bulk_ * (J - 1.0) * (J - 1.0));
}

void NeoHookeanMaterial::EvalStress(const mfem::DenseMatrix& F, mfem::DenseMatrix& sigma) const
{
  int dim = F.Width();
  B_.SetSize(dim);
  sigma.SetSize(dim);

  if (c_mu_) {
    EvalCoeffs();
  }

  // See http://solidmechanics.org/Text/Chapter3_5/Chapter3_5.php for a sample derivation
  double dJ = F.Det();

  double a = mu_ * std::pow(dJ, -(2.0 / dim) - 1.0);
  double b = bulk_ * (dJ - 1.0) - a * (F * F) / dim;

  mfem::MultABt(F, F, B_);

  sigma = 0.0;

  sigma.Add(a, B_);

  for (int i = 0; i < dim; ++i) {
    sigma(i, i) += b;
  }
}

void NeoHookeanMaterial::AssembleTangentModuli(const mfem::DenseMatrix& F, mfem_ext::Array4D<double>& C) const
{
  int dim = F.Width();
  B_.SetSize(dim);
  C.SetSize(dim, dim, dim, dim);

  mfem::MultABt(F, F, B_);

  double dJ = F.Det();

  if (c_mu_) {
    EvalCoeffs();
  }

  // See http://solidmechanics.org/Text/Chapter8_4/Chapter8_4.php for a sample derivation
  double a = mu_ * std::pow(dJ, -2.0 / dim);
  double b = bulk_ * (2.0 * dJ - 1.0) * dJ + a * (2.0 / (dim * dim)) * (F * F);

  C = 0.0;

  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      for (int k = 0; k < dim; ++k) {
        for (int l = 0; l < dim; ++l) {
          if (i == k) {
            C(i, j, k, l) += a * B_(j, l);
          }
          if (j == k) {
            C(i, j, k, l) += a * B_(i, l);
          }
          if (k == l) {
            C(i, j, k, l) -= a * (2.0 / dim) * B_(i, j);
          }
          if (i == j) {
            C(i, j, k, l) -= a * (2.0 / dim) * B_(k, l);
          }
          if (i == j && k == l) {
            C(i, j, k, l) += b;
          }
        }
      }
    }
  }
}

inline void LinearElasticMaterial::EvalCoeffs() const
{
  mu_   = c_mu_->Eval(*Ttr_, Ttr_->GetIntPoint());
  bulk_ = c_bulk_->Eval(*Ttr_, Ttr_->GetIntPoint());
}

void LinearElasticMaterial::EvalStress(const mfem::DenseMatrix& F, mfem::DenseMatrix& sigma) const
{
  int dim = F.Width();
  sigma.SetSize(dim);
  epsilon_.SetSize(dim);
  FT_.SetSize(dim);

  if (c_mu_) {
    EvalCoeffs();
  }

  // Evaluate the linearized strain tensor from the deformation gradient
  epsilon_ = 0.0;
  epsilon_.Add(0.5, F);
  FT_.Transpose(F);
  epsilon_.Add(0.5, FT_);

  for (int i = 0; i < dim; ++i) {
    epsilon_(i, i) -= 1.0;
  }

  sigma                = 0.0;
  double trace_epsilon = epsilon_.Trace();

  // Calculate the stress by Hooke's law
  sigma.Add(2.0 * mu_, epsilon_);
  for (int i = 0; i < dim; ++i) {
    sigma(i, i) += bulk_ * trace_epsilon - (2.0 / dim) * mu_ * trace_epsilon;
  }
}

void LinearElasticMaterial::AssembleTangentModuli(const mfem::DenseMatrix& F, mfem_ext::Array4D<double>& C) const
{
  int dim = F.Width();

  if (c_mu_) {
    EvalCoeffs();
  }

  C = 0.0;

  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      for (int k = 0; k < dim; ++k) {
        for (int l = 0; l < dim; ++l) {
          if (i == j && k == l) {
            C(i, j, k, l) += bulk_ - (2.0 / dim) * mu_;
          }
          if (i == k && j == l) {
            C(i, j, k, l) += mu_;
          }
          if (i == l && j == k) {
            C(i, j, k, l) += mu_;
          }
        }
      }
    }
  }
}

}  // namespace serac
// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/materials/hyperelastic_material.hpp"

#include "serac/infrastructure/logger.hpp"

#include <cmath>

namespace serac {

void HyperelasticMaterial::calcDeformationGradient(const mfem::DenseMatrix& du_dX, mfem::DenseMatrix& F)
{
  int dim = du_dX.Size();
  F.SetSize(dim);
  F = du_dX;

  for (int i = 0; i < dim; ++i) {
    F(i, i) += 1.0;
  }
}

void HyperelasticMaterial::calcLinearizedStrain(const mfem::DenseMatrix& du_dX, mfem::DenseMatrix& epsilon)
{
  epsilon.SetSize(du_dX.Size());
  epsilon = du_dX;
  epsilon.Symmetrize();
}

void HyperelasticMaterial::calcCauchyStressFromPK1Stress(const mfem::DenseMatrix& F, const mfem::DenseMatrix& P,
                                                         mfem::DenseMatrix& sigma)
{
  sigma.SetSize(F.Size());
  mfem::MultABt(P, F, sigma);
  sigma *= 1.0 / F.Det();
}

inline void NeoHookeanMaterial::EvalCoeffs() const
{
  mu_   = c_mu_->Eval(*parent_to_reference_transformation_, parent_to_reference_transformation_->GetIntPoint());
  bulk_ = c_bulk_->Eval(*parent_to_reference_transformation_, parent_to_reference_transformation_->GetIntPoint());
}

double NeoHookeanMaterial::evalStrainEnergy(const mfem::DenseMatrix& du_dX) const
{
  calcDeformationGradient(du_dX, F_);

  int dim = F_.Width();

  SLIC_ERROR_IF(dim != 2 && dim != 3, "NeoHookean material used for spatial dimension not 2 or 3!");

  EvalCoeffs();

  double det_J = F_.Det();

  // First invariant of FF^T
  double I1_bar = pow(det_J, -2.0 / dim) * (F_ * F_);

  return 0.5 * (mu_ * (I1_bar - dim) + bulk_ * (det_J - 1.0) * (det_J - 1.0));
}

void NeoHookeanMaterial::evalStress(const mfem::DenseMatrix& du_dX, mfem::DenseMatrix& sigma) const
{
  calcDeformationGradient(du_dX, F_);
  int dim = F_.Width();
  B_.SetSize(dim);
  sigma.SetSize(dim);

  EvalCoeffs();

  // See http://solidmechanics.org/Text/Chapter3_5/Chapter3_5.php for a sample derivation
  double det_J = F_.Det();

  double a = mu_ * std::pow(det_J, -(2.0 / dim) - 1.0);
  double b = bulk_ * (det_J - 1.0) - a * (F_ * F_) / dim;

  // Form the left Cauchy-Green deformation tensor
  mfem::MultABt(F_, F_, B_);

  sigma = 0.0;

  sigma.Add(a, B_);

  for (int i = 0; i < dim; ++i) {
    sigma(i, i) += b;
  }
}

void NeoHookeanMaterial::evalTangentStiffness(const mfem::DenseMatrix& du_dX, mfem_ext::Array4D<double>& C) const
{
  calcDeformationGradient(du_dX, F_);
  int dim = F_.Width();
  B_.SetSize(dim);
  C.SetSize(dim, dim, dim, dim);

  mfem::MultABt(F_, F_, B_);

  double det_J = F_.Det();

  EvalCoeffs();

  // See http://solidmechanics.org/Text/Chapter8_4/Chapter8_4.php for a sample derivation
  double a = mu_ * std::pow(det_J, -2.0 / dim);
  double b = bulk_ * (2.0 * det_J - 1.0) * det_J + a * (2.0 / (dim * dim)) * (F_ * F_);

  C = 0.0;

  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      for (int k = 0; k < dim; ++k) {
        for (int l = 0; l < dim; ++l) {
          C(i, j, k, l) += a * (B_(j, l) * (i == k) + B_(i, l) * (j == k)) -
                           a * (2.0 / dim) * (B_(i, j) * (k == l) + B_(k, l) * (i == j)) + b * (i == j) * (k == l);
        }
      }
    }
  }
}

inline void LinearElasticMaterial::EvalCoeffs() const
{
  mu_   = c_mu_->Eval(*parent_to_reference_transformation_, parent_to_reference_transformation_->GetIntPoint());
  bulk_ = c_bulk_->Eval(*parent_to_reference_transformation_, parent_to_reference_transformation_->GetIntPoint());
}

void LinearElasticMaterial::evalStress(const mfem::DenseMatrix& du_dX, mfem::DenseMatrix& sigma) const
{
  int dim = du_dX.Width();
  sigma.SetSize(dim);
  epsilon_.SetSize(dim);

  EvalCoeffs();

  // Evaluate the linearized strain tensor from the displacement gradient
  calcLinearizedStrain(du_dX, epsilon_);

  sigma                = 0.0;
  double trace_epsilon = epsilon_.Trace();

  // Calculate the stress by Hooke's law
  sigma.Add(2.0 * mu_, epsilon_);
  for (int i = 0; i < dim; ++i) {
    sigma(i, i) += bulk_ * trace_epsilon - (2.0 / dim) * mu_ * trace_epsilon;
  }
}

void LinearElasticMaterial::evalTangentStiffness(const mfem::DenseMatrix& du_dX, mfem_ext::Array4D<double>& C) const
{
  int dim = du_dX.Width();

  EvalCoeffs();

  C = 0.0;

  double lambda = bulk_ - (2.0 / dim) * mu_;

  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      for (int k = 0; k < dim; ++k) {
        for (int l = 0; l < dim; ++l) {
          C(i, j, k, l) += lambda * (i == j && k == l) + mu_ * ((i == k && j == l) + (i == l && j == k));
        }
      }
    }
  }
}

}  // namespace serac

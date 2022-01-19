// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/coefficients/sensitivity_coefficients.hpp"
#include "serac/physics/materials/solid_utils.hpp"

namespace serac::mfem_ext {

ShearSensitivityCoefficient::ShearSensitivityCoefficient(FiniteElementState&    displacement,
                                                         FiniteElementState&    adjoint_displacement,
                                                         LinearElasticMaterial& linear_mat)
    : displacement_(displacement), adjoint_displacement_(adjoint_displacement), material_(linear_mat)
{
  dim_ = displacement_.gridFunc().ParFESpace()->GetVDim();

  du_dX_.SetSize(dim_);
  dp_dX_.SetSize(dim_);
  adjoint_strain_.SetSize(dim_);
  d_sigma_d_shear_.SetSize(dim_);
}

double ShearSensitivityCoefficient::Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip)
{
  T.SetIntPoint(&ip);

  // Compute the displacement and deformation gradient
  displacement_.gridFunc().GetVectorGradient(T, du_dX_);

  // Compute the adjoint gradient and strain
  adjoint_displacement_.gridFunc().GetVectorGradient(T, dp_dX_);

  serac::solid_util::calcLinearizedStrain(dp_dX_, adjoint_strain_);

  // Evaluate the derivative of the Cauchy stress with respect to the shear modulus using the calculated deformation
  // gradient
  material_.EvalShearSensitivity(du_dX_, d_sigma_d_shear_);

  double scalar_sum = 0.0;

  // Compute the inner product of the stress sensitivity and the adjoint strain
  for (int i = 0; i < dim_; ++i) {
    for (int j = 0; j < dim_; ++j) {
      scalar_sum -= d_sigma_d_shear_(i, j) * adjoint_strain_(i, j);
    }
  }

  return scalar_sum;
}

BulkSensitivityCoefficient::BulkSensitivityCoefficient(FiniteElementState&    displacement,
                                                       FiniteElementState&    adjoint_displacement,
                                                       LinearElasticMaterial& linear_mat)
    : displacement_(displacement), adjoint_displacement_(adjoint_displacement), material_(linear_mat)
{
  dim_ = displacement_.gridFunc().ParFESpace()->GetVDim();

  du_dX_.SetSize(dim_);
  dp_dX_.SetSize(dim_);
  adjoint_strain_.SetSize(dim_);
  d_sigma_d_bulk_.SetSize(dim_);
}

double BulkSensitivityCoefficient::Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip)
{
  T.SetIntPoint(&ip);

  // Compute the displacement and deformation gradient
  displacement_.gridFunc().GetVectorGradient(T, du_dX_);

  // Compute the adjoint gradient and strain
  adjoint_displacement_.gridFunc().GetVectorGradient(T, dp_dX_);

  serac::solid_util::calcLinearizedStrain(dp_dX_, adjoint_strain_);

  // Evaluate the derivative of the Cauchy stress with respect to the bulk modulus using the calculated deformation
  // gradient
  material_.EvalBulkSensitivity(du_dX_, d_sigma_d_bulk_);

  double scalar_sum = 0.0;

  // Compute the inner product of the stress sensitivity and the adjoint strain
  for (int i = 0; i < dim_; ++i) {
    for (int j = 0; j < dim_; ++j) {
      scalar_sum -= d_sigma_d_bulk_(i, j) * adjoint_strain_(i, j);
    }
  }

  return scalar_sum;
}

}  // namespace serac::mfem_ext

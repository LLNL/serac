// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/coefficients/sensitivity_coefficients.hpp"
#include "serac/physics/utilities/physics_utils.hpp"

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
  F_.SetSize(dim_);
}

double ShearSensitivityCoefficient::Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip)
{
  T.SetIntPoint(&ip);

  displacement_.gridFunc().GetVectorGradient(T, du_dX_);

  solid_util::calcDeformationGradient(du_dX_, F_);

  double det_J = F_.Det();

  adjoint_displacement_.gridFunc().GetVectorGradient(T, dp_dX_);

  serac::solid_util::calcLinearizedStrain(dp_dX_, adjoint_strain_);

  // Evaluate the Cauchy stress using the calculated deformation gradient
  material_.EvalShearSensitivity(du_dX_, d_sigma_d_shear_);

  // Accumulate the residual using the Cauchy stress and the B matrix
  d_sigma_d_shear_ *= det_J * ip.weight * T.Weight();

  double scalar_sum = 0.0;

  for (int i = 0; i < dim_; ++i) {
    for (int j = 0; j < dim_; ++j) {
      scalar_sum += d_sigma_d_shear_(i, j) * adjoint_strain_(i, j);
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
  F_.SetSize(dim_);
}

double BulkSensitivityCoefficient::Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip)
{
  T.SetIntPoint(&ip);

  displacement_.gridFunc().GetVectorGradient(T, du_dX_);

  solid_util::calcDeformationGradient(du_dX_, F_);

  double det_J = F_.Det();

  adjoint_displacement_.gridFunc().GetVectorGradient(T, dp_dX_);

  serac::solid_util::calcLinearizedStrain(dp_dX_, adjoint_strain_);

  // Evaluate the Cauchy stress using the calculated deformation gradient
  material_.EvalBulkSensitivity(du_dX_, d_sigma_d_bulk_);

  // Accumulate the residual using the Cauchy stress and the B matrix
  d_sigma_d_bulk_ *= det_J * ip.weight * T.Weight();

  double scalar_sum = 0.0;

  for (int i = 0; i < dim_; ++i) {
    for (int j = 0; j < dim_; ++j) {
      scalar_sum += d_sigma_d_bulk_(i, j) * adjoint_strain_(i, j);
    }
  }

  return scalar_sum;
}

}  // namespace serac::mfem_ext

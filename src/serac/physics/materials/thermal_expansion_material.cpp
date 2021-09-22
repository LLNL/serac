// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/materials/thermal_expansion_material.hpp"
#include "serac/physics/utilities/physics_utils.hpp"

#include "serac/infrastructure/logger.hpp"

#include <cmath>

namespace serac {

inline void IsotropicThermalExpansionMaterial::EvalCoeffs() const
{
  coef_thermal_expansion_ = c_coef_thermal_expansion_->Eval(*parent_to_reference_transformation_,
                                                            parent_to_reference_transformation_->GetIntPoint());
  reference_temp_ =
      c_reference_temp_->Eval(*parent_to_reference_transformation_, parent_to_reference_transformation_->GetIntPoint());
  temp_ = temp_state_.gridFuncCoef().Eval(*parent_to_reference_transformation_,
                                          parent_to_reference_transformation_->GetIntPoint());
}

void IsotropicThermalExpansionMaterial::modifyDisplacementGradient(mfem::DenseMatrix& du_dX)
{
  double expansion = coef_thermal_expansion_ * (temp_ - reference_temp_);

  for (int i = 0; i < du_dX.Width(); ++i) {
    du_dX(i, i) += expansion * (1.0 + du_dX(i, i));
  }
}

void IsotropicThermalExpansionMaterial::modifyDeformationGradient(mfem::DenseMatrix& F)
{
  EvalCoeffs();

  F_copy_ = F;

  F_thermal_.SetSize(F.Width());
  F_thermal_ = 0.0;

  for (int i = 0; i < F.Width(); ++i) {
    F_thermal_(i, i) = 1.0 + coef_thermal_expansion_ * (temp_ - reference_temp_);
  }

  mfem::Mult(F_copy_, F_thermal_, F);
}

void IsotropicThermalExpansionMaterial::modifyLinearizedStrain(mfem::DenseMatrix& epsilon)
{
  EvalCoeffs();

  for (int i = 0; i < epsilon.Width(); ++i) {
    epsilon(i, i) += coef_thermal_expansion_ * (temp_ - reference_temp_);
  }
}

}  // namespace serac

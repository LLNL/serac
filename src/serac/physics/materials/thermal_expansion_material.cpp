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

void IsotropicThermalExpansionMaterial::modifyDisplacementGradient(mfem::DenseMatrix& du_dX)
{
  auto coef_thermal_expansion = c_coef_thermal_expansion_->Eval(*parent_to_reference_transformation_,
                                                                parent_to_reference_transformation_->GetIntPoint());
  auto reference_temp =
      c_reference_temp_->Eval(*parent_to_reference_transformation_, parent_to_reference_transformation_->GetIntPoint());
  auto current_temp = temp_state_.gridFuncCoef().Eval(*parent_to_reference_transformation_,
                                                      parent_to_reference_transformation_->GetIntPoint());

  auto expansion = coef_thermal_expansion * (reference_temp - current_temp);

  for (int i = 0; i < du_dX.Width(); ++i) {
    du_dX(i, i) += expansion;

    // If the geometric nonlinearities are turned on, consider the extra expansion
    // in the current configuration
    if (geom_nonlin_ == GeometricNonlinearities::On) {
      du_dX(i, i) += expansion * du_dX(i, i);
    }
  }
}

}  // namespace serac

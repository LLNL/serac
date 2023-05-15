// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/thermomechanics_input.hpp"

namespace serac {

void ThermomechanicsInputOptions::defineInputFileSchema(axom::inlet::Container& container)
{
  // The solid mechanics options
  auto& solid_solver_table = container.addStruct("solid", "Finite deformation solid mechanics module").required();
  serac::SolidMechanicsInputOptions::defineInputFileSchema(solid_solver_table);

  // The thermal conduction options
  auto& thermal_solver_table = container.addStruct("thermal_conduction", "Thermal conduction module").required();
  serac::HeatTransferInputOptions::defineInputFileSchema(thermal_solver_table);

  auto& ref_temp = container.addStruct("reference_temperature",
                                       "Coefficient for the reference temperature for isotropic thermal expansion");
  serac::input::CoefficientInputOptions::defineInputFileSchema(ref_temp);

  auto& coef_therm_expansion =
      container.addStruct("coef_thermal_expansion", "Coefficient of thermal expansion for isotropic thermal expansion");
  serac::input::CoefficientInputOptions::defineInputFileSchema(coef_therm_expansion);

  container.registerVerifier([](const axom::inlet::Container& base) -> bool {
    bool cte_found      = base.contains("coef_thermal_expansion");
    bool ref_temp_found = base.contains("reference_temperature");

    if (ref_temp_found && cte_found) {
      return true;
    } else if ((!ref_temp_found) && (!cte_found)) {
      return true;
    }

    SLIC_WARNING_ROOT(
        "Either both a coefficient of thermal expansion and reference temperature should be specified"
        "in the thermal solid input file or neither should be.");

    return false;
  });
}

}  // namespace serac

serac::ThermomechanicsInputOptions FromInlet<serac::ThermomechanicsInputOptions>::operator()(
    const axom::inlet::Container& base)
{
  serac::ThermomechanicsInputOptions result;

  result.solid_options = base["solid"].get<serac::SolidMechanicsInputOptions>();

  result.thermal_options = base["thermal_conduction"].get<serac::HeatTransferInputOptions>();

  if (base.contains("coef_thermal_expansion")) {
    result.coef_thermal_expansion = base["coef_thermal_expansion"].get<serac::input::CoefficientInputOptions>();
    result.reference_temperature  = base["reference_temperature"].get<serac::input::CoefficientInputOptions>();
  }

  return result;
}

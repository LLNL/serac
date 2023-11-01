// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/materials/thermal_material_input.hpp"

namespace serac {

void ThermalMaterialInputOptions::defineInputFileSchema(axom::inlet::Container& container)
{
  // Define schema with each thermal material parameter
  container.addString("model", "The model of material").required(true);
  container.addDouble("density", "Initial mass density");
  container.addDouble("kappa", "The conductivity parameter");
  container.addDouble("cp", "The specific heat capacity");

  // Verify
  container.registerVerifier([](const axom::inlet::Container& c) -> bool {
    std::string model = c["model"];
    if (model == "LinearIsotropicConductor") {
      return true;  // TODO
    } else if (model == "LinearConductor") {
      return true;  // TODO
    }

    return false;
  });
}

}  // namespace serac

serac::var_thermal_material_t FromInlet<serac::var_thermal_material_t>::operator()(const axom::inlet::Container& base)
{
  serac::var_thermal_material_t result;
  std::string                   model = base["model"];

  if (model == "LinearIsotropicConductor") {
    result = serac::heat_transfer::LinearIsotropicConductor(base["density"], base["cp"], base["kappa"]);
  } else if (model == "LinearConductor") {
    // TODO
  }

  return result;
}

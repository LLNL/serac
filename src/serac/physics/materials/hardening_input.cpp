// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/materials/solid_material_input.hpp"

namespace serac {

void HardeningInputOptions::defineInputFileSchema(axom::inlet::Container& container)
{
  // Shared between both hardening laws
  container.addString("law", "Name of the hardening law").required(true);
  container.addDouble("sigma_y", "Yield strength");

  // PowerLawHardening
  container.addDouble("n", "Hardening index in reciprocal form");
  container.addDouble("eps0", "Reference value of accumulated plastic strain");

  // VoceHardening
  container.addDouble("sigma_sat", "Saturation value of flow strength");
  container.addDouble("strain_constant", "Constant dictating how fast the exponential decays");
}

}  // namespace serac

serac::var_hardening_t FromInlet<serac::var_hardening_t>::operator()(const axom::inlet::Container& base)
{
  serac::var_hardening_t result;
  std::string law = base["law"];
  if (law == "PowerLawHardening") {
    result = serac::solid_mechanics::PowerLawHardening{.sigma_y = base["sigma_y"],
                                                       .n = base["n"],
                                                       .eps0 = base["eps0"]};
  }
  else if (law == "VoceHardening") {
    result = serac::solid_mechanics::VoceHardening{.sigma_y = base["sigma_y"],
                                                       .sigma_sat = base["sigma_sat"],
                                                       .strain_constant = base["strain_constant"]};
  }
  return result;
}

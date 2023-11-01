// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/materials/solid_material_input.hpp"

namespace serac {

void SolidMaterialInputOptions::defineInputFileSchema(axom::inlet::Container& container)
{
  // Define schema with each solid material parameter
  container.addString("model", "The model of material").required(true);
  container.addDouble("density", "Initial mass density");

  // Solid mechanics (neo-hookean, linear isotropic)
  container.addDouble("mu", "The shear modulus");
  container.addDouble("K", "The bulk modulus");

  // Solid mechanics (j2, j2nonlinear)
  container.addDouble("E", "Young's modulus");
  container.addDouble("nu", "Poisson's ratio");
  container.addDouble("Hi", "Isotropic hardening constant");
  container.addDouble("Hk", "Kinematic hardening constant");
  container.addDouble("sigma_y", "Yield stress");

  // Verify
  container.registerVerifier([](const axom::inlet::Container& c) -> bool {
    std::string model = c["model"];
    if (model == "NeoHookean") {
      return true;  // TODO
    } else if (model == "LinearIsotropic") {
      return true;  // TODO
    } else if (model == "J2") {
      return true;  // TODO
    } else if (model == "J2Nonlinear") {
      return true;  // TODO
    }

    return false;
  });
}

}  // namespace serac

serac::var_solid_material_t FromInlet<serac::var_solid_material_t>::operator()(const axom::inlet::Container& base)
{
  serac::var_solid_material_t result;
  std::string                 model = base["model"];

  if (model == "NeoHookean") {
    result = serac::solid_mechanics::NeoHookean{.density = base["density"], .K = base["K"], .G = base["mu"]};
  } else if (model == "LinearIsotropic") {
    result = serac::solid_mechanics::LinearIsotropic{.density = base["density"], .K = base["K"], .G = base["mu"]};
  } else if (model == "J2") {
    // result = serac::solid_mechanics::J2{.E = base["E"],
    //                                     .nu = base["nu"],
    //                                     .Hi = base["Hi"],
    //                                     .Hk = base["Hk"],
    //                                     .sigma_y = base["sigma_y"],
    //                                     .density = base["density"]};
  } else if (model == "J2Nonlinear") {
    // TODO create a hardening material input hpp/ cpp
    // serac::solid_mechanics::PowerLawHardening hardening{.sigma_y = 1.0, .n = 1.0, .eps0 = 1.0};
    // result = serac::solid_mechanics::J2Nonlinear<serac::solid_mechanics::PowerLawHardening>{.E = base["E"],
    //                                           .nu = base["nu"],
    //                                           .hardening = hardening,
    //                                           .density = base["density"]};
  }

  return result;
}

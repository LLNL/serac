// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/materials/material_input.hpp"

namespace serac {

void MaterialInputOptions::defineInputFileSchema(axom::inlet::Container& container)
{
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

  // Heat transfer parameters
  container.addDouble("kappa", "The conductivity parameter");
  container.addDouble("cp", "The specific heat capacity");
}

void MaterialInputOptions::verifyInputFileSchema(axom::inlet::Container& container)
{
  container.registerVerifier([](const axom::inlet::Container& container) -> bool {
    bool model_present = container.contains("model") && (container["model"].type() == axom::inlet::InletType::String);
    bool density_present =
        container.contains("density") && (container["density"].type() == axom::inlet::InletType::Double);

    // Solid mechanics (neo-hookean, linear isotropic)
    bool K_present  = container.contains("K") && (container["K"].type() == axom::inlet::InletType::Double);
    bool mu_present = container.contains("mu") && (container["mu"].type() == axom::inlet::InletType::Double);

    // Solid mechanics (j2, j2nonlinear)
    // bool E_present = container.contains("E") &&
    //   (container["E"].type() == axom::inlet::InletType::Double);
    // bool nu_present = container.contains("nu") &&
    //   (container["nu"].type() == axom::inlet::InletType::Double);
    // bool Hi_present = container.contains("Hi") &&
    //   (container["Hi"].type() == axom::inlet::InletType::Double);
    // bool Hk_present = container.contains("Hk") &&
    //   (container["Hk"].type() == axom::inlet::InletType::Double);
    // bool sigma_y_present = container.contains("sigma_y") &&
    //   (container["sigma_y"].type() == axom::inlet::InletType::Double);

    // Heat transfer
    bool kappa_present = container.contains("kappa") && (container["kappa"].type() == axom::inlet::InletType::Double);
    bool cp_present    = container.contains("cp") && (container["cp"].type() == axom::inlet::InletType::Double);

    if (!model_present) return false;
    std::string mat = container["model"];

    // TODO, also check if invalid variables are not present?
    if (mat == "NeoHookean" || mat == "LinearIsotropic") {
      return density_present && K_present && mu_present;
    } else if (mat == "J2") {
      // TODO
    } else if (mat == "J2Nonlinear") {
      // TODO
    } else if (mat == "LinearIsotropicConductor") {
      return density_present && kappa_present && cp_present;
    } else if (mat == "LinearConductor") {
      // TODO
    }

    return false;
  });
}

}  // namespace serac

serac::MaterialInputOptions FromInlet<serac::MaterialInputOptions>::operator()(const axom::inlet::Container& base)
{
  serac::MaterialInputOptions result;

  result.model   = base["model"];
  result.density = base["density"];

  // Solid mechanics (neo-hookean, linear isotropic)
  result.mu = base["mu"];
  result.K  = base["K"];

  // Solid mechanics (j2, j2nonlinear)
  result.E       = base["E"];
  result.nu      = base["nu"];
  result.Hi      = base["Hi"];
  result.Hk      = base["Hk"];
  result.sigma_y = base["sigma_y"];

  // Heat transfer
  result.kappa = base["kappa"];
  result.cp    = base["cp"];

  return result;
}

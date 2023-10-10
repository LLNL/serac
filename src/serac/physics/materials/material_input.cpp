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
  container.addDouble("density", "Initial mass density").defaultValue(1.0);

  // Solid mechanics (neo-hookean, linear isotropic)
  container.addDouble("mu", "The shear modulus").defaultValue(0.25);
  container.addDouble("K", "The bulk modulus").defaultValue(5.0);

  // Solid mechanics (j2, j2nonlinear)
  container.addDouble("E", "Young's modulus").defaultValue(1.0);                // TODO default value
  container.addDouble("nu", "Poisson's ratio").defaultValue(1.0);               // TODO default value
  container.addDouble("Hi", "Isotropic hardening constant").defaultValue(1.0);  // TODO default value
  container.addDouble("Hk", "Kinematic hardening constant").defaultValue(1.0);  // TODO default value
  container.addDouble("sigma_y", "Yield stress").defaultValue(1.0);             // TODO default value

  // Heat transfer parameters
  container.addDouble("kappa", "The conductivity parameter").defaultValue(1.0);
  container.addDouble("cp", "The specific heat capacity").defaultValue(1.0);
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
  result.mu      = base["mu"];
  result.K       = base["K"];
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

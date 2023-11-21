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
  container.addDouble("cp", "The specific heat capacity");

  // For LinearIsotropicConductor
  container.addDouble("kappa", "The conductivity parameter");

  // For LinearConductor
  container.addInt("dim", "Dimension of conductivity tensor parameter");
  container.addDoubleArray("kappa_tensor", "The conductivity tensor parameter");  // TODO make 2d-array

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
    // Store tensor in a vector temporarily
    std::vector<double> v = base["kappa_tensor"].get<std::vector<double>>();

    // Set the tensor values and material result based on the dimension
    int dim = base["dim"];
    if (dim == 2) {
      serac::tensor<double, 2, 2> cond = {{{v[0], v[1]}, {v[2], v[3]}}};
      result                           = serac::heat_transfer::LinearConductor<2>(base["density"], base["cp"], cond);
    } else if (dim == 3) {
      serac::tensor<double, 3, 3> cond = {{{v[0], v[1], v[2]}, {v[3], v[4], v[5]}, {v[6], v[7], v[8]}}};
      result                           = serac::heat_transfer::LinearConductor<3>(base["density"], base["cp"], cond);
    }
  }

  return result;
}

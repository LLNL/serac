// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file material_driver.hpp
 *
 * @brief Utility for testing material model output
 */

#pragma once

#include <iostream>

#include "serac/physics/materials/solid_functional_material.hpp"
#include "serac/numerics/functional/tensor.hpp"

namespace serac::solid_util {

template <typename T>
class MaterialDriver {
 public:

  MaterialDriver(const T& material):
      material_(material)
  {
    // empty
  };

  void run_uniaxial(double maxEngineeringStrain, int nsteps)
  {
    const double strain_increment = maxEngineeringStrain/nsteps;
    const tensor<double, 3> X{};
    const tensor<double, 3> u{};
    tensor<double, 3, 3> dudx{};
    
    for (int i = 0; i < nsteps; i++) {
      dudx[0][0] += strain_increment;
      dudx[1][1] -= 0.25*strain_increment;
      dudx[2][2] -= 0.25*strain_increment;
      auto [density, stress] = material_(X, u, dudx);
      std::cout << "step " << i << " stress " << stress[0][0] << std::endl;
    }
  }

 private:
  const T& material_;
};

} // namespace serac

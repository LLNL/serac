// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file test_material_point_driver.cpp
 *
 * @brief unit tests for the material point test utility
 */

#include <gtest/gtest.h>

#include "serac/physics/materials/material_driver.hpp"
#include "serac/physics/materials/solid_functional_material.hpp"

namespace serac {

TEST(MaterialDriver, testUniaxialTensionCaseAgainstClosedFormSolution)
{
  double density = 1.0;
  double E = 1.0;
  double nu = 0.25;
  double G = 0.5*E/(1.0 + nu);
  double K = E/3.0/(1.0 - 2.0*nu);
  solid_util::LinearIsotropicSolid<3> material(density, G, K);
  solid_util::MaterialDriver material_driver(material);
  double max_strain = 1.0;
  int steps = 10;
  material_driver.run_uniaxial(max_strain, steps);
}

} // namespace serac

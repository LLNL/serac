// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file nonlinear_J2_material.cpp
 */

#include "serac/physics/materials/solid_material.hpp"

#include <iostream>
#include <fstream>

#include <gtest/gtest.h>

#include "serac/numerics/functional/tensor.hpp"

namespace serac {

TEST(NonlinearJ2Material, Hardening)
{
    solid_mechanics::Hardening hardening_law{.sigma_y = 1.0, .n=2.0, .eps0=0.01};
    std::ofstream file;
    file.open("stress.csv", std::ios::in | std::ios::trunc);

    double eqps = 0.0;
    for (size_t i = 0; i < 50; i++) {
        auto stress = hardening_law(make_dual(eqps));
        file << eqps << " " << stress.value << " " << stress.gradient << std::endl;
        eqps += 0.01;
    }
};

} // namespace serac


int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();

  return result;
}
// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>
#include "serac/infrastructure/input.hpp"

namespace serac {

namespace test_utils {

void defineNonlinSolidInputFileSchema(axom::inlet::Inlet& inlet);

void runNonlinSolidTest(const std::string& input_file);

void defineThermalConductionInputFileSchema(axom::inlet::Inlet& inlet);

void runThermalConductionTest(const std::string& input_file);

class InputFileTest : public ::testing::TestWithParam<std::string> {
};

}  // end namespace test_utils

}  // end namespace serac

// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>
#include "serac/infrastructure/input.hpp"

namespace serac {

namespace test_utils {

template <typename PhysicsModule>
void defineTestSchema(axom::inlet::Inlet& inlet);

template <typename PhysicsModule>
void runModuleTest(const std::string& input_file, std::shared_ptr<mfem::ParMesh> custom_mesh = {});

class InputFileTest : public ::testing::TestWithParam<std::string> {
};

}  // end namespace test_utils

}  // end namespace serac

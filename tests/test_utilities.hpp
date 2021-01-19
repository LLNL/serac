// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include <gtest/gtest.h>
#include "serac/infrastructure/input.hpp"

namespace serac::test_utils {

/**
 * @brief Defines the full schema for an integration test on the provided
 * Inlet object
 */
template <typename PhysicsModule>
void defineTestSchema(axom::inlet::Inlet& inlet);

/**
 * @brief Runs a simulation and checks the output fields if "correct" answers
 * were defined in the input file
 * @param[in] input_file The Lua input file
 * @param[in] custom_mesh Overrides the mesh described in the input file
 */
template <typename PhysicsModule>
void runModuleTest(const std::string& input_file, std::unique_ptr<mfem::ParMesh> custom_mesh = {});

class InputFileTest : public ::testing::TestWithParam<std::string> {
};

}  // end namespace serac::test_utils

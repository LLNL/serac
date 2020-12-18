// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/infrastructure/input.hpp"

namespace serac {

namespace test_utils {

void defineSolidInputFileSchema(axom::inlet::Inlet& inlet);

void runSolidTest(const std::string& input_file);

}  // end namespace test_utils

}  // end namespace serac

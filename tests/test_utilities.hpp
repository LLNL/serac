// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/infrastructure/input.hpp"

namespace serac {

namespace testing {

void defineNonlinSolidInputFileSchema(axom::inlet::Inlet& inlet);

void runNonlinSolidDynamicTest(const std::string& input_file);

}  // end namespace testing

}  // end namespace serac

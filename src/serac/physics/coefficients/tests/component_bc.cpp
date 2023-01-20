// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/physics/coefficients/coefficient_extensions.hpp"
#include "serac/infrastructure/input.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/serac_config.hpp"
#include "serac/physics/solid_legacy.hpp"
#include "serac/physics/tests/test_utilities.hpp"

namespace serac {

TEST(SolidLegacy, QsAttributeSolve)
{
  MPI_Barrier(MPI_COMM_WORLD);
  std::string input_file_path = std::string(SERAC_REPO_DIR) + "/data/input_files/tests/solid/qs_attribute_solve.lua";
  test_utils::runModuleTest<SolidLegacy>(input_file_path, "qs_attribute_solve");
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(SolidLegacy, QsComponentSolve)
{
  MPI_Barrier(MPI_COMM_WORLD);
  std::string input_file_path = std::string(SERAC_REPO_DIR) + "/data/input_files/tests/solid/qs_component_solve.lua";
  test_utils::runModuleTest<SolidLegacy>(input_file_path, "qs_component_solve");
  MPI_Barrier(MPI_COMM_WORLD);
}

}  // namespace serac

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}

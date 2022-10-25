// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <sys/stat.h>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/thermal_solid_legacy.hpp"
#include "serac/serac_config.hpp"
#include "test_utilities.hpp"

namespace serac {

using test_utils::InputFileTest;

TEST_P(InputFileTest, ThermalSolidLegacy)
{
  MPI_Barrier(MPI_COMM_WORLD);
  std::string input_file_path =
      std::string(SERAC_REPO_DIR) + "/data/input_files/tests/thermal_solid/" + GetParam() + ".lua";
  test_utils::runModuleTest<ThermalSolidLegacy>(input_file_path, GetParam());
  MPI_Barrier(MPI_COMM_WORLD);
}

const std::string input_files[] = {"thermal_expansion"};

INSTANTIATE_TEST_SUITE_P(ThermalSolidLegacyInputFileTest, InputFileTest, ::testing::ValuesIn(input_files));

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

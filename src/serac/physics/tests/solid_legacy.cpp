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
#include "serac/physics/solid_legacy.hpp"
#include "serac/serac_config.hpp"
#include "test_utilities.hpp"

namespace serac {

using test_utils::InputFileTest;

TEST_P(InputFileTest, SolidLegacy)
{
  MPI_Barrier(MPI_COMM_WORLD);
  std::string input_file_path = std::string(SERAC_REPO_DIR) + "/data/input_files/tests/solid/" + GetParam() + ".lua";
  test_utils::runModuleTest<SolidLegacy>(input_file_path, GetParam());
  MPI_Barrier(MPI_COMM_WORLD);
}

const std::string input_files[] = {"dyn_solve", "dyn_direct_solve",
// TODO Disabled while we diagnose the non-deterministic sundials error
/*
#ifdef MFEM_USE_SUNDIALS
                                   "dyn_linesearch_solve",
#endif
*/
#ifdef MFEM_USE_AMGX
                                   "dyn_amgx_solve",
#endif
                                   "qs_solve",
                                   // "qs_direct_solve", disabled due to segfault in DSUPERLU
                                   "qs_linear"};

INSTANTIATE_TEST_SUITE_P(SolidLegacyInputFileTest, InputFileTest, ::testing::ValuesIn(input_files));

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

// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include "serac/physics/thermal_conduction.hpp"
#include "serac/serac_config.hpp"
#include "tests/test_utilities.hpp"

using namespace serac;

TEST(thermal_solver, dyn_imp_solve)
{
  // Start a scope block to guarantee separation between the simulated nominal/restart runs
  {
    MPI_Barrier(MPI_COMM_WORLD);
    const std::string input_file_path =
        std::string(SERAC_REPO_DIR) + "/data/input_files/tests/thermal_conduction/dyn_imp_solve.lua";
    test_utils::runModuleTest<ThermalConduction>(input_file_path);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  serac::StateManager::reset();

  // Simulate a restart
  {
    MPI_Barrier(MPI_COMM_WORLD);
    const std::string input_file_path =
        std::string(SERAC_REPO_DIR) + "/data/input_files/tests/thermal_conduction/dyn_imp_solve_restart.lua";
    const int restart_cycle = 5;
    test_utils::runModuleTest<ThermalConduction>(input_file_path, {}, restart_cycle);
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

//------------------------------------------------------------------------------
#include "axom/slic/core/SimpleLogger.hpp"

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;  // create & initialize test logger, finalized when
                                    // exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}

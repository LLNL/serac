// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/thermal_conduction.hpp"

#include <sys/stat.h>

#include <fstream>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/serac_config.hpp"
#include "test_utilities.hpp"

namespace serac {

using test_utils::InputFileTest;

TEST_P(InputFileTest, thermal_conduction)
{
  MPI_Barrier(MPI_COMM_WORLD);
  std::string input_file_path =
      std::string(SERAC_REPO_DIR) + "/data/input_files/tests/thermal_conduction/" + GetParam() + ".lua";
  test_utils::runModuleTest<ThermalConduction>(input_file_path, GetParam());
  MPI_Barrier(MPI_COMM_WORLD);
}

const std::string input_files[] = {"static_solve",
#ifdef MFEM_USE_AMGX
                                   "static_amgx_solve",
#endif
                                   "static_solve_multiple_bcs",
                                   "static_solve_repeated_bcs",
                                   "static_reaction_exact",
                                   "dyn_exp_solve",
                                   "dyn_imp_solve",
                                   "dyn_imp_solve_time_varying"};

INSTANTIATE_TEST_SUITE_P(ThermalConductionInputFileTests, InputFileTest, ::testing::ValuesIn(input_files));

TEST(thermal_solver, dyn_imp_solve_restart)
{
  // Start a scope block to guarantee separation between the simulated nominal/restart runs
  {
    MPI_Barrier(MPI_COMM_WORLD);
    const std::string input_file_path =
        std::string(SERAC_REPO_DIR) + "/data/input_files/tests/thermal_conduction/dyn_imp_solve.lua";
    test_utils::runModuleTest<ThermalConduction>(input_file_path, "dyn_imp_solve_restart");
    MPI_Barrier(MPI_COMM_WORLD);
  }

  serac::StateManager::reset();

  // Simulate a restart
  {
    MPI_Barrier(MPI_COMM_WORLD);
    const std::string input_file_path =
        std::string(SERAC_REPO_DIR) + "/data/input_files/tests/thermal_conduction/dyn_imp_solve_restart.lua";
    const int restart_cycle = 5;
    test_utils::runModuleTest<ThermalConduction>(input_file_path, "dyn_imp_solve_restart", restart_cycle);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  serac::StateManager::reset();
}

}  // namespace serac

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

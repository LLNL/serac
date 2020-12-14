// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include "serac/serac_config.hpp"
#include "test_utilities.hpp"

namespace serac {

TEST(dynamic_solver, dyn_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);
  std::string input_file_path =
      std::string(SERAC_REPO_DIR) + "/data/input_files/tests/nonlinear_solid/serac_dynamic_solver/dyn_solve.lua";
  testing::runNonlinSolidDynamicTest(input_file_path);
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(dynamic_solver, dyn_direct_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);
  std::string input_file_path =
      std::string(SERAC_REPO_DIR) + "/data/input_files/tests/nonlinear_solid/serac_dynamic_solver/dyn_direct_solve.lua";
  testing::runNonlinSolidDynamicTest(input_file_path);
  MPI_Barrier(MPI_COMM_WORLD);
}

#ifdef MFEM_USE_SUNDIALS
TEST(dynamic_solver, dyn_linesearch_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);
  std::string input_file_path = std::string(SERAC_REPO_DIR) +
                                "/data/input_files/tests/nonlinear_solid/serac_dynamic_solver/dyn_linesearch_solve.lua";
  testing::runNonlinSolidDynamicTest(input_file_path);
  MPI_Barrier(MPI_COMM_WORLD);
}
#endif  // MFEM_USE_SUNDIALS

#ifdef MFEM_USE_AMGX
TEST(dynamic_solver, dyn_amgx_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);
  std::string input_file_path =
      std::string(SERAC_REPO_DIR) + "/data/input_files/tests/nonlinear_solid/serac_dynamic_solver/dyn_amgx_solve.lua";
  testing::runNonlinSolidDynamicTest(input_file_path);
  MPI_Barrier(MPI_COMM_WORLD);
}
#endif  // MFEM_USE_AMGX

}  // namespace serac

//------------------------------------------------------------------------------
#include "axom/slic/core/UnitTestLogger.hpp"
using axom::slic::UnitTestLogger;

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  UnitTestLogger logger;  // create & initialize test logger, finalized when exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}

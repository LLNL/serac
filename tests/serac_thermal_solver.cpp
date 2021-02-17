// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/thermal_conduction.hpp"

#include <sys/stat.h>

#include <fstream>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/numerics/mesh_utils.hpp"
#include "serac/serac_config.hpp"
#include "test_utilities.hpp"

namespace serac {

using test_utils::InputFileTest;

double One(const mfem::Vector& /*x*/) { return 1.0; }
double BoundaryTemperature(const mfem::Vector& x) { return x.Norml2(); }
double OtherBoundaryTemperature(const mfem::Vector& x) { return 2 * x.Norml2(); }

double InitialTemperature(const mfem::Vector& x)
{
  if (x.Norml2() < 0.5) {
    return 2.0;
  } else {
    return 1.0;
  }
}

TEST(thermal_solver, static_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);
  std::string input_file_path =
      std::string(SERAC_REPO_DIR) + "/data/input_files/tests/thermal_conduction/static_solve.lua";
  auto pmesh = buildBallMesh(10000);
  test_utils::runModuleTest<ThermalConduction>(input_file_path, pmesh);
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_P(InputFileTest, thermal_conduction)
{
  MPI_Barrier(MPI_COMM_WORLD);
  std::string input_file_path =
      std::string(SERAC_REPO_DIR) + "/data/input_files/tests/thermal_conduction/" + GetParam() + ".lua";
  test_utils::runModuleTest<ThermalConduction>(input_file_path);
  MPI_Barrier(MPI_COMM_WORLD);
}

const std::string input_files[] = {"static_solve_multiple_bcs", "static_solve_repeated_bcs", "dyn_exp_solve",
                                   "dyn_imp_solve", "dyn_imp_solve_time_varying"};

INSTANTIATE_TEST_SUITE_P(ThermalConductionInputFileTests, InputFileTest, ::testing::ValuesIn(input_files));

#ifdef MFEM_USE_AMGX
TEST(thermal_solver, static_amgx_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);
  std::string input_file_path =
      std::string(SERAC_REPO_DIR) + "/data/input_files/tests/thermal_conduction/static_amgx_solve.lua";
  auto pmesh = buildBallMesh(10000);
  test_utils::runModuleTest<ThermalConduction>(input_file_path, pmesh);
  MPI_Barrier(MPI_COMM_WORLD);
}
#endif

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

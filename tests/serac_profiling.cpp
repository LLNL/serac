// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <exception>

#include <gtest/gtest.h>

#include "serac/infrastructure/cli.hpp"
#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/profiling.hpp"
#include "serac/numerics/mesh_utils.hpp"
#include <cstring>

namespace serac {

TEST(serac_profiling, mesh_refinement)
{
  // profile mesh refinement
  MPI_Barrier(MPI_COMM_WORLD);
  serac::profiling::initializeCaliper();

  std::string mesh_file = std::string(SERAC_REPO_DIR) + "/data/meshes/bortel_echem.e";

  SERAC_MARK_START("RefineAndLoadMesh");
  auto pmesh = mesh::refineAndDistribute(SERAC_PROFILE_EXPR("LOAD_MESH", buildMeshFromFile(mesh_file)), 0, 0);
  SERAC_MARK_END("RefineAndLoadMesh");

  SERAC_MARK_LOOP_START(refinement_loop, "refinement_loop");
  for (int i = 0; i < 2; i++) {
    SERAC_MARK_LOOP_ITER(refinement_loop, i);
    pmesh->UniformRefinement();
  }
  SERAC_MARK_LOOP_END(refinement_loop);

  // Refine once more and utilize SERAC_PROFILE_SCOPE
  {
    SERAC_PROFILE_SCOPE("RefineOnceMore");
    pmesh->UniformRefinement();
  }

  SERAC_SET_METADATA("mesh_file", mesh_file.c_str());
  SERAC_SET_METADATA("number_mesh_elements", pmesh->GetNE());

  // this number represents "llnl" as an unsigned integer
  unsigned int magic_uint = 1819176044;
  SERAC_SET_METADATA("magic_uint", magic_uint);

  // decode unsigned int back into char[4]
  std::array<char, sizeof(magic_uint) + 1> uint_string;
  std::fill(std::begin(uint_string), std::end(uint_string), 0);
  std::memcpy(uint_string.data(), &magic_uint, 4);
  std::cout << uint_string.data() << std::endl;

  // encode double with "llnl" bytes
  double magic_double;
  std::memcpy(&magic_double, "llnl", 4);
  SERAC_SET_METADATA("magic_double", magic_double);

  // decode the double and print
  std::array<char, sizeof(magic_double) + 1> double_string;
  std::fill(std::begin(double_string), std::end(double_string), 0);
  std::memcpy(double_string.data(), &magic_double, 4);
  std::cout << double_string.data() << std::endl;

  serac::profiling::terminateCaliper();

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(serac_profiling, exception)
{
  // profile mesh refinement
  MPI_Barrier(MPI_COMM_WORLD);
  serac::profiling::initializeCaliper();

  {
    SERAC_PROFILE_SCOPE("Non-exceptionScope");
    try {
      SERAC_PROFILE_SCOPE("Exception scope");
      throw std::runtime_error("Caliper to verify RAII");
    } catch (std::exception& e) {
      std::cout << e.what() << "\n";
    }
  }

  serac::profiling::terminateCaliper();

  MPI_Barrier(MPI_COMM_WORLD);
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

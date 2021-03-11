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

  SERAC_MARK_START("LOAD_MESH");
		     
  std::string mesh_file = std::string(SERAC_REPO_DIR) + "/data/meshes/bortel_echem.e";

  SERAC_MARK_END("LOAD_MESH");

  auto pmesh = mesh::refineAndDistribute(buildMeshFromFile(mesh_file), 0, 0);

  SERAC_MARK_LOOP_START(refinement_loop, "refinement_loop");
  for (int i = 0; i < 2 ; i++)
    {
      SERAC_MARK_LOOP_ITER(refinement_loop, i);
      pmesh->UniformRefinement();
    }
  SERAC_MARK_LOOP_END(refinement_loop);
  
  serac::profiling::setCaliperMetadata("mesh_file", mesh_file.c_str());
  serac::profiling::setCaliperMetadata("number_mesh_elements", pmesh->GetNE());

  unsigned int magic_uint = 1819176044;
  serac::profiling::setCaliperMetadata("magic_uint", magic_uint);

  char uint_string[sizeof(magic_uint)+1];
  std::memcpy(uint_string, &magic_uint, sizeof(magic_uint));
  std::cout << uint_string << std::endl;
  
  double magic_double;
  std::memcpy(&magic_double, "llnl", 4);
  serac::profiling::setCaliperMetadata("magic_double", magic_double);

  std::array<char, sizeof(magic_double) + 1> double_string;
  std::memcpy(double_string.data(), &magic_double, sizeof(magic_double));
  std::cout << double_string.data() << std::endl;
  
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

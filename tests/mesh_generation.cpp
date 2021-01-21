// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/numerics/mesh_utils.hpp"

#include <gtest/gtest.h>

TEST(meshgen, successful_creation)
{
  // the disk and ball meshes don't exactly hit the number
  // of elements specified, they refine to get as close as possible
  ASSERT_EQ(serac::buildDiskMesh(1000)->GetNE(), 1024);
  ASSERT_EQ(serac::buildBallMesh(6000)->GetNE(), 4096);
  ASSERT_EQ(serac::buildRectangleMesh(20, 20)->GetNE(), 400);
  ASSERT_EQ(serac::buildCuboidMesh(20, 20, 20)->GetNE(), 8000);
}

//------------------------------------------------------------------------------
#include "axom/slic/core/UnitTestLogger.hpp"
using axom::slic::UnitTestLogger;

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  UnitTestLogger logger;  // create & initialize test logger, finalized when
                          // exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}

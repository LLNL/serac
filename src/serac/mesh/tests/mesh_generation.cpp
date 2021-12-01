// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/mesh/mesh_utils.hpp"
#include "serac/serac_config.hpp"
#include <gtest/gtest.h>
#include <exception>

class SlicErrorException : public std::exception {
};

TEST(meshgen, successful_creation)
{
  // the disk and ball meshes don't exactly hit the number
  // of elements specified, they refine to get as close as possible
  ASSERT_EQ(serac::buildDiskMesh(1000).GetNE(), 1024);
  ASSERT_EQ(serac::buildBallMesh(6000).GetNE(), 4096);
  ASSERT_EQ(serac::buildRectangleMesh(20, 20, 1., 2.).GetNE(), 400);
  ASSERT_EQ(serac::buildCuboidMesh(20, 20, 20, 1., 2., 3.).GetNE(), 8000);
  ASSERT_EQ(serac::buildCylinderMesh(2, 2, 2.0, 5.0).GetNE(), 384);
  ASSERT_EQ(serac::buildHollowCylinderMesh(2, 2, 2.0, 3.0, 5.0).GetNE(), 256);
  ASSERT_EQ(serac::buildHollowCylinderMesh(2, 2, 2.0, 3.0, 5.0, 2. * M_PI, 8).GetNE(), 256);
  ASSERT_EQ(serac::buildHollowCylinderMesh(2, 2, 2.0, 3.0, 5.0, 2. * M_PI, 6).GetNE(), 192);
  ASSERT_EQ(serac::buildHollowCylinderMesh(2, 1, 2.0, 3.0, 5.0, M_PI / 3., 1).GetNE(), 16);
  ASSERT_EQ(serac::buildHollowCylinderMesh(2, 1, 2.0, 3.0, 5.0, 2. * M_PI, 7).GetNE(), 112);
}

TEST(meshgen, lua_input)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  axom::sidre::DataStore datastore;

  // Initialize Inlet and read input file
  std::string input_file = std::string(SERAC_REPO_DIR) + "/data/input_files/tests/meshing/meshing.lua";
  std::cout << input_file << std::endl;
  auto inlet = serac::input::initialize(datastore, input_file);

  auto& mesh_file_table = inlet.addStruct("main_mesh_from_file", "A mesh to build from file");
  serac::mesh::InputOptions::defineInputFileSchema(mesh_file_table);

  auto& mesh_cuboid_table = inlet.addStruct("main_mesh_cuboid", "A cuboid mesh");
  serac::mesh::InputOptions::defineInputFileSchema(mesh_cuboid_table);

  auto& mesh_rect_table = inlet.addStruct("main_mesh_rect", "A rectangular mesh");
  serac::mesh::InputOptions::defineInputFileSchema(mesh_rect_table);

  // Verify input file
  if (!inlet.verify()) {
    SLIC_ERROR("Input file failed to verify.");
  }

  // temporary scope to build mesh from file
  {
    auto       mesh_options = inlet["main_mesh_from_file"].get<serac::mesh::InputOptions>();
    const auto file_options = std::get_if<serac::mesh::FileInputOptions>(&mesh_options.extra_options);
    ASSERT_NE(file_options, nullptr);
    auto full_mesh_path = serac::input::findMeshFilePath(file_options->relative_mesh_file_name, input_file);
    file_options->absolute_mesh_file_name =
        serac::input::findMeshFilePath(file_options->relative_mesh_file_name, input_file);
    auto mesh = serac::mesh::buildParallelMesh(mesh_options);
  }

  // temporary scope to build a cuboid mesh
  {
    auto       mesh_options   = inlet["main_mesh_cuboid"].get<serac::mesh::InputOptions>();
    const auto cuboid_options = std::get_if<serac::mesh::BoxInputOptions>(&mesh_options.extra_options);
    ASSERT_NE(cuboid_options, nullptr);
    EXPECT_EQ(cuboid_options->elements.size(), 3);
    auto mesh = serac::mesh::buildParallelMesh(mesh_options);
    EXPECT_EQ(mesh->GetNE(), cuboid_options->elements[0] * cuboid_options->elements[1] * cuboid_options->elements[2]);
  }

  // temporary scope to build a rectangular mesh
  {
    auto       mesh_options = inlet["main_mesh_rect"].get<serac::mesh::InputOptions>();
    const auto rect_options = std::get_if<serac::mesh::BoxInputOptions>(&mesh_options.extra_options);
    ASSERT_NE(rect_options, nullptr);
    EXPECT_EQ(rect_options->elements.size(), 2);
    auto mesh = serac::mesh::buildParallelMesh(mesh_options);
    EXPECT_EQ(mesh->GetNE(), rect_options->elements[0] * rect_options->elements[1]);
  }

  // Check that we fail on an invalid mesh description
  auto& mesh_fail_table = inlet.addStruct("main_mesh_fail", "An invalid mesh description");
  serac::mesh::InputOptions::defineInputFileSchema(mesh_fail_table);
  SLIC_INFO("Begin expected warning about invalid value.");
  EXPECT_FALSE(inlet.verify());
  SLIC_INFO("End expected warning about invalid value.");


  MPI_Barrier(MPI_COMM_WORLD);
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

  axom::slic::setAbortFunction([]() { throw SlicErrorException{}; });
  axom::slic::setAbortOnError(true);
  axom::slic::setAbortOnWarning(false);

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}

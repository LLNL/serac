// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/numerics/mesh_utils.hpp"
#include "serac/serac_config.hpp"
#include <gtest/gtest.h>
#include <exception>

class SlicErrorException : public std::exception {
};

TEST(meshgen, successful_creation)
{
  // the disk and ball meshes don't exactly hit the number
  // of elements specified, they refine to get as close as possible
  ASSERT_EQ(serac::buildDiskMesh(1000)->GetNE(), 1024);
  ASSERT_EQ(serac::buildBallMesh(6000)->GetNE(), 4096);
  ASSERT_EQ(serac::buildRectangleMesh(20, 20)->GetNE(), 400);
  ASSERT_EQ(serac::buildCuboidMesh(20, 20, 20)->GetNE(), 8000);
  ASSERT_EQ(serac::buildRectangleMesh(20, 20, 1., 2.)->GetNE(), 400);
  ASSERT_EQ(serac::buildCuboidMesh(20, 20, 20, 1., 2., 3.)->GetNE(), 8000);
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

  auto& mesh_file_table = inlet.addTable("main_mesh_from_file", "A mesh to build from file");
  serac::mesh::InputOptions::defineInputFileSchema(mesh_file_table);

  auto& mesh_cuboid_table = inlet.addTable("main_mesh_cuboid", "A cuboid mesh");
  serac::mesh::InputOptions::defineInputFileSchema(mesh_cuboid_table);

  auto& mesh_rect_table = inlet.addTable("main_mesh_rect", "A rectangular mesh");
  serac::mesh::InputOptions::defineInputFileSchema(mesh_rect_table);

  auto& mesh_fail_table = inlet.addTable("main_mesh_fail", "An invalid mesh description");
  serac::mesh::InputOptions::defineInputFileSchema(mesh_fail_table);

  // Verify input file
  if (!inlet.verify()) {
    SLIC_ERROR("Input file failed to verify.");
  }

  // temporary scope to build mesh from file
  {
    auto mesh_options = inlet["main_mesh_from_file"].get<serac::mesh::InputOptions>();
    if (const auto file_options = std::get_if<serac::mesh::FileInputOptions>(&mesh_options.extra_options)) {
      auto full_mesh_path = serac::input::findMeshFilePath(file_options->relative_mesh_file_name, input_file);
      auto mesh = serac::buildMeshFromFile(full_mesh_path, mesh_options.ser_ref_levels, mesh_options.par_ref_levels);
    }
  }

  // temporary scope to build a cuboid mesh
  {
    auto mesh_options = inlet["main_mesh_cuboid"].get<serac::mesh::InputOptions>();
    if (const auto cuboid_options = std::get_if<serac::mesh::GenerateInputOptions>(&mesh_options.extra_options)) {
      EXPECT_EQ(cuboid_options->elements.size(), 3);
      auto mesh = serac::buildCuboidMesh(*cuboid_options);
      EXPECT_EQ(mesh->GetNE(), cuboid_options->elements[0] * cuboid_options->elements[1] * cuboid_options->elements[2]);
    }
  }

  // temporary scope to build a rectangular mesh
  {
    auto mesh_options = inlet["main_mesh_rect"].get<serac::mesh::InputOptions>();
    if (const auto rect_options = std::get_if<serac::mesh::GenerateInputOptions>(&mesh_options.extra_options)) {
      EXPECT_EQ(rect_options->elements.size(), 2);
      auto mesh = serac::buildRectangleMesh(*rect_options);
      EXPECT_EQ(mesh->GetNE(), rect_options->elements[0] * rect_options->elements[1]);
    }
  }

  // temporary scope to build a rectangular mesh
  {
    EXPECT_THROW(auto mesh_options = inlet["main_mesh_fail"].get<serac::mesh::InputOptions>(), SlicErrorException);
  }

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

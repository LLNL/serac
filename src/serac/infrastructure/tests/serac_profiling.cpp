// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <array>
#include <cstring>
#include <exception>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>

#include "serac/infrastructure/cli.hpp"
#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/profiling.hpp"
#include "serac/mesh/mesh_utils.hpp"

namespace serac {

TEST(SeracProfiling, MeshRefinement)
{
  // profile mesh refinement
  MPI_Barrier(MPI_COMM_WORLD);
  serac::profiling::initialize();

  std::string mesh_file = std::string(SERAC_REPO_DIR) + "/data/meshes/bortel_echem.e";

  // the following string is a proxy for templated test names
  std::string test_name = "_profiling";

  SERAC_MARK_BEGIN(profiling::concat("RefineAndLoadMesh", test_name).c_str());
  auto pmesh = mesh::refineAndDistribute(SERAC_PROFILE_EXPR("LOAD_MESH", buildMeshFromFile(mesh_file)), 0, 0);
  SERAC_MARK_END(profiling::concat("RefineAndLoadMesh", test_name).c_str());

  SERAC_MARK_LOOP_BEGIN(refinement_loop, "refinement_loop");
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

  serac::profiling::finalize();

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(SeracProfiling, Exception)
{
  // profile mesh refinement
  MPI_Barrier(MPI_COMM_WORLD);
  serac::profiling::initialize();

  {
    SERAC_PROFILE_SCOPE("Non-exceptionScope");
    try {
      SERAC_PROFILE_SCOPE("Exception scope");
      throw std::runtime_error("Caliper to verify RAII");
    } catch (std::exception& e) {
      std::cout << e.what() << "\n";
    }
  }

  serac::profiling::finalize();

  MPI_Barrier(MPI_COMM_WORLD);
}

struct NonCopyableOrMovable {
  int value                                         = 0;
  NonCopyableOrMovable()                            = default;
  NonCopyableOrMovable(const NonCopyableOrMovable&) = delete;
  NonCopyableOrMovable(NonCopyableOrMovable&&)      = delete;
};

TEST(SeracProfiling, LvalueReferenceExpr)
{
  MPI_Barrier(MPI_COMM_WORLD);
  serac::profiling::initialize();
  NonCopyableOrMovable foo;
  // This statement requires that the RHS be *exactly* a non-const lvalue
  // reference - of course a const lvalue reference cannot be bound here,
  // but also an rvalue reference would also cause compilation to fail
  NonCopyableOrMovable& bar = SERAC_PROFILE_EXPR("lvalue_reference_assign", foo);
  serac::profiling::finalize();
  bar.value = 6;
  EXPECT_EQ(foo.value, 6);
  MPI_Barrier(MPI_COMM_WORLD);
}

struct MovableOnly {
  int value                       = 0;
  MovableOnly()                   = default;
  MovableOnly(const MovableOnly&) = delete;
  MovableOnly(MovableOnly&&)      = default;
};

TEST(SeracProfiling, RvalueReferenceExpr)
{
  MPI_Barrier(MPI_COMM_WORLD);
  serac::profiling::initialize();
  MovableOnly foo;
  foo.value = 6;
  // This statement requires that the RHS be *exactly* an rvalue reference
  // An lvalue reference cannot be used to construct here (copy ctor deleted)
  MovableOnly bar = SERAC_PROFILE_EXPR("rvalue_reference_assign", std::move(foo));
  serac::profiling::finalize();
  EXPECT_EQ(bar.value, 6);
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(SeracProfiling, TempRvalueReferenceExpr)
{
  MPI_Barrier(MPI_COMM_WORLD);
  serac::profiling::initialize();
  // This statement requires that the RHS be *exactly* an rvalue reference
  // An lvalue reference cannot be used to construct here (copy ctor deleted)
  MovableOnly bar = SERAC_PROFILE_EXPR("rvalue_reference_assign", MovableOnly{6});
  serac::profiling::finalize();
  EXPECT_EQ(bar.value, 6);
  MPI_Barrier(MPI_COMM_WORLD);
}

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

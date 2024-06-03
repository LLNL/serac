// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
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

#if defined(SERAC_USE_CALIPER) && defined(SERAC_USE_ADIAK)

namespace serac {

TEST(Profiling, MeshRefinement)
{
  // profile mesh refinement
  CALI_CXX_MARK_FUNCTION;
  MPI_Barrier(MPI_COMM_WORLD);
  serac::profiling::initialize();

  std::string mesh_file = std::string(SERAC_REPO_DIR) + "/data/meshes/bortel_echem.e";

  // the following string is a proxy for templated test names
  std::string test_name = "_profiling";

  CALI_MARK_BEGIN(profiling::concat("RefineAndLoadMesh", test_name).c_str());
  auto pmesh = mesh::refineAndDistribute(SERAC_PROFILE_EXPR("LOAD_MESH", buildMeshFromFile(mesh_file)), 0, 0);
  CALI_MARK_END(profiling::concat("RefineAndLoadMesh", test_name).c_str());

  CALI_CXX_MARK_LOOP_BEGIN(refinement_loop, "refinement_loop");
  for (int i = 0; i < 2; i++) {
    CALI_CXX_MARK_LOOP_ITERATION(refinement_loop, i);
    pmesh->UniformRefinement();
  }
  CALI_CXX_MARK_LOOP_END(refinement_loop);

  // Refine once more and utilize CALI_CXX_MARK_SCOPE
  {
    CALI_CXX_MARK_SCOPE("RefineOnceMore");
    pmesh->UniformRefinement();
  }

  adiak::value("mesh_file", mesh_file.c_str());
  adiak::value("number_mesh_elements", pmesh->GetNE());

  // this number represents "llnl" as an unsigned integer
  unsigned int magic_uint = 1819176044;
  adiak::value("magic_uint", magic_uint);

  // decode unsigned int back into char[4]
  std::array<char, sizeof(magic_uint) + 1> uint_string;
  std::fill(std::begin(uint_string), std::end(uint_string), 0);
  std::memcpy(uint_string.data(), &magic_uint, 4);
  std::cout << uint_string.data() << std::endl;

  // encode double with "llnl" bytes
  double magic_double;
  std::memcpy(&magic_double, "llnl", 4);
  adiak::value("magic_double", magic_double);

  // decode the double and print
  std::array<char, sizeof(magic_double) + 1> double_string;
  std::fill(std::begin(double_string), std::end(double_string), 0);
  std::memcpy(double_string.data(), &magic_double, 4);
  std::cout << double_string.data() << std::endl;

  serac::profiling::finalize();

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(Profiling, Exception)
{
  // profile mesh refinement
  MPI_Barrier(MPI_COMM_WORLD);
  serac::profiling::initialize();

  {
    CALI_CXX_MARK_SCOPE("Non-exceptionScope");
    try {
      CALI_CXX_MARK_SCOPE("Exception scope");
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

TEST(Profiling, LvalueReferenceExpr)
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

TEST(Profiling, RvalueReferenceExpr)
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

TEST(Profiling, TempRvalueReferenceExpr)
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

#endif

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

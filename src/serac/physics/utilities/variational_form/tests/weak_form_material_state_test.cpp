// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include "serac/serac_config.hpp"

#include "serac/numerics/mesh_utils.hpp"

#include "serac/infrastructure/initialize.hpp"
// #include "serac/infrastructure/input.hpp"
// #include "serac/infrastructure/logger.hpp"
#include "serac/infrastructure/terminator.hpp"

#include "serac/physics/utilities/variational_form/weak_form.hpp"
#include "serac/physics/utilities/variational_form/tensor.hpp"

#include "serac/physics/utilities/finite_element_state.hpp"

using namespace serac;

struct State {
  double x;
};

bool operator==(const State& lhs, const State& rhs) { return lhs.x == rhs.x; }

class QuadratureDataTest : public ::testing::Test {
protected:
  void SetUp() override
  {
    constexpr auto mesh_file = SERAC_REPO_DIR "/data/meshes/star.mesh";
    mesh                     = mesh::refineAndDistribute(buildMeshFromFile(mesh_file), 0, 0);
    festate                  = std::make_unique<FiniteElementState>(*mesh);
    festate->gridFunc()      = 0.0;
    residual = std::make_unique<WeakForm<test_space(trial_space)>>(&festate->space(), &festate->space());
  }
  static constexpr int p   = 1;
  static constexpr int dim = 2;
  using test_space         = H1<p>;
  using trial_space        = H1<p>;
  std::unique_ptr<mfem::ParMesh>                     mesh;
  std::unique_ptr<FiniteElementState>                festate;
  std::unique_ptr<WeakForm<test_space(trial_space)>> residual;
};

TEST_F(QuadratureDataTest, basic_integrals)
{
  QuadratureData<State> qdata(*mesh, p);
  State                 init{0};
  qdata = init;
  residual->AddDomainIntegral(
      Dimension<dim>{},
      [&](auto /* x */, auto u, auto& state) {
        // std::cout << "a\n";
        state.x += 0.1;
        return u;
      },
      *mesh, qdata);
  // If we run through it one time...
  (*residual)(festate->gridFunc());
  // Then each element of the state should have been incremented accordingly...
  State correct{0.1};
  for (const auto& s : qdata.data()) {
    EXPECT_EQ(s, correct);
  }
  // Just to make sure I didn't break the stateless version
  residual->AddSurfaceIntegral([&](auto x, auto /* u */) { return x[0]; }, *mesh);
}

//------------------------------------------------------------------------------
#include "axom/slic/core/SimpleLogger.hpp"

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;  // create & initialize test logger, finalized when exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}

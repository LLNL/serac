// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include "serac/serac_config.hpp"

#include "serac/numerics/mesh_utils.hpp"

#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/terminator.hpp"

#include "serac/physics/utilities/functional/functional.hpp"
#include "serac/physics/utilities/functional/tensor.hpp"

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
    residual = std::make_unique<Functional<test_space(trial_space)>>(&festate->space(), &festate->space());
  }
  static constexpr int p   = 1;
  static constexpr int dim = 2;
  using test_space         = H1<p>;
  using trial_space        = H1<p>;
  std::unique_ptr<mfem::ParMesh>                       mesh;
  std::unique_ptr<FiniteElementState>                  festate;
  std::unique_ptr<Functional<test_space(trial_space)>> residual;
};

TEST_F(QuadratureDataTest, basic_integrals)
{
  QuadratureData<State> qdata(*mesh, p);
  State                 init{0.1};
  qdata = init;
  residual->AddDomainIntegral(
      Dimension<dim>{},
      [&](auto /* x */, auto u, auto& state) {
        state.x += 0.1;
        return u;
      },
      *mesh, qdata);

  // If we run through it one time...
  mfem::Vector U(festate->space().TrueVSize());
  (*residual)(U);
  // Then each element of the state should have been incremented accordingly...
  State correct{0.2};
  for (const auto& s : qdata) {
    EXPECT_EQ(s, correct);
  }
  // Just to make sure I didn't break the stateless version
  residual->AddDomainIntegral(
      Dimension<dim>{}, [&](auto /* x */, auto u) { return u; }, *mesh);
}

struct StateWithDefault {
  double x = 0.5;
};

bool operator==(const StateWithDefault& lhs, const StateWithDefault& rhs) { return lhs.x == rhs.x; }

TEST_F(QuadratureDataTest, basic_integrals_default)
{
  QuadratureData<StateWithDefault> qdata(*mesh, p);
  residual->AddDomainIntegral(
      Dimension<dim>{},
      [&](auto /* x */, auto u, auto& state) {
        state.x += 0.1;
        return u;
      },
      *mesh, qdata);
  // If we run through it one time...
  mfem::Vector U(festate->space().TrueVSize());
  (*residual)(U);
  // Then each element of the state should have been incremented accordingly...
  StateWithDefault correct{0.6};
  const auto&      const_qdata = qdata;
  for (auto& s : const_qdata) {
    EXPECT_EQ(s, correct);
  }
}

struct StateWithMultiFields {
  double x = 0.5;
  double y = 0.3;
};

bool operator==(const StateWithMultiFields& lhs, const StateWithMultiFields& rhs)
{
  return lhs.x == rhs.x && lhs.y == rhs.y;
}

TEST_F(QuadratureDataTest, basic_integrals_multi_fields)
{
  QuadratureData<StateWithMultiFields> qdata(*mesh, p);
  residual->AddDomainIntegral(
      Dimension<dim>{},
      [&](auto /* x */, auto u, auto& state) {
        state.x += 0.1;
        state.y += 0.7;
        return u;
      },
      *mesh, qdata);
  // If we run through it one time...
  mfem::Vector U(festate->space().TrueVSize());
  (*residual)(U);
  // Then each element of the state should have been incremented accordingly...
  StateWithMultiFields correct{0.6, 1.0};
  for (const auto& s : qdata) {
    EXPECT_EQ(s, correct);
  }
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

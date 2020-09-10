// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include <memory>

#include "common/common.hpp"
#include "mfem.hpp"

namespace serac {

TEST(boundary_cond, simple_repeated_dofs)
{
  MPI_Barrier(MPI_COMM_WORLD);
  constexpr int      N    = 15;
  constexpr int      ATTR = 1;
  mfem::Mesh         mesh(N, N, mfem::Element::TRIANGLE);
  mfem::ParMesh      par_mesh(MPI_COMM_WORLD, mesh);
  FiniteElementState state(par_mesh);

  for (int i = 0; i < par_mesh.GetNBE(); i++) {
    par_mesh.GetBdrElement(i)->SetAttribute(ATTR);
  }

  BoundaryConditionManager bcs(par_mesh);
  auto                     coef = std::make_shared<mfem::ConstantCoefficient>(1);
  bcs.addEssential({ATTR}, coef, state, 1);
  const auto before_dofs = bcs.allDofs();

  bcs.addEssential({ATTR}, coef, state, 1);
  const auto after_dofs = bcs.allDofs();

  // Make sure that attempting to add a boundary condition
  // on already-used elements doesn't change the dofs
  EXPECT_EQ(before_dofs.Size(), after_dofs.Size());
  MPI_Barrier(MPI_COMM_WORLD);
}

enum TestTag
{
  Tag1,
  Tag2
};

TEST(boundary_cond, filter_generics)
{
  MPI_Barrier(MPI_COMM_WORLD);
  constexpr int N = 15;
  mfem::Mesh    mesh(N, N, mfem::Element::TRIANGLE);
  mfem::ParMesh par_mesh(MPI_COMM_WORLD, mesh);

  BoundaryConditionManager bcs(par_mesh);
  auto                     coef = std::make_shared<mfem::ConstantCoefficient>(1);
  for (int i = 0; i < N; i++) {
    bcs.addGeneric({}, coef, TestTag::Tag1, 1);
    bcs.addGeneric({}, coef, TestTag::Tag2, 1);
  }

  int bcs_with_tag1 = 0;
  for (const auto& bc : bcs.genericsWithTag(TestTag::Tag1)) {
    EXPECT_EQ(TestTag::Tag1, bc.tag());
    bcs_with_tag1++;
  }
  EXPECT_EQ(bcs_with_tag1, N);

  int bcs_with_tag2 = 0;
  for (const auto& bc : bcs.genericsWithTag(TestTag::Tag2)) {
    EXPECT_EQ(TestTag::Tag2, bc.tag());
    bcs_with_tag2++;
  }
  EXPECT_EQ(bcs_with_tag2, N);

  MPI_Barrier(MPI_COMM_WORLD);
}

}  // namespace serac

//------------------------------------------------------------------------------
#include "axom/slic/core/UnitTestLogger.hpp"
using axom::slic::UnitTestLogger;

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  UnitTestLogger logger;  // create & initialize test logger, finalized when exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}

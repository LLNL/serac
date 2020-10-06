// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include <memory>

#include "mfem.hpp"
#include "physics/utilities/boundary_condition_manager.hpp"

namespace serac {

struct TestTag {
  static constexpr bool should_be_scalar = false;
};

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

  using TestDirichlet = StrongAlias<EssentialBoundaryCondition, TestTag>;
  BoundaryConditionManager<TestDirichlet> bcs(par_mesh);
  auto                                    coef = std::make_shared<mfem::ConstantCoefficient>(1);
  bcs.addEssential<TestTag>({ATTR}, coef, 1);
  for (auto& bc : bcs.essentials<TestTag>()) {
    bc.setTrueDofs(state);
  }
  const auto before_dofs = bcs.allEssentialDofs<TestTag>();

  bcs.addEssential<TestTag>({ATTR}, coef, 1);
  for (auto& bc : bcs.essentials<TestTag>()) {
    bc.setTrueDofs(state);
  }
  const auto after_dofs = bcs.allEssentialDofs<TestTag>();

  // Make sure that attempting to add a boundary condition
  // on already-used elements doesn't change the dofs
  EXPECT_EQ(before_dofs.Size(), after_dofs.Size());
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

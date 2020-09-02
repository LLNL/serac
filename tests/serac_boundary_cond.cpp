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

  for (int i = 0; i < par_mesh.GetNE(); i++) {
    par_mesh.SetAttribute(i, ATTR);
  }

  BoundaryConditionManager bcs;
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

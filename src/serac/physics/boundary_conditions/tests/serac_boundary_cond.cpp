// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "serac/physics/boundary_conditions/boundary_condition_helper.hpp"

#include <memory>

#include <gtest/gtest.h>
#include "mfem.hpp"

namespace serac {

TEST(boundary_cond, simple_repeated_dofs)
{
  MPI_Barrier(MPI_COMM_WORLD);
  constexpr int      N    = 15;
  constexpr int      ATTR = 1;
  auto               mesh = mfem::Mesh::MakeCartesian2D(N, N, mfem::Element::TRIANGLE);
  mfem::ParMesh      par_mesh(MPI_COMM_WORLD, mesh);
  FiniteElementState state(par_mesh);

  for (int i = 0; i < par_mesh.GetNBE(); i++) {
    par_mesh.GetBdrElement(i)->SetAttribute(ATTR);
  }

  BoundaryConditionManager bcs(par_mesh);
  auto                     coef = std::make_shared<mfem::ConstantCoefficient>(1);
  bcs.addEssential({ATTR}, coef, state, 1);
  const auto before_dofs = bcs.allEssentialDofs();

  bcs.addEssential({ATTR}, coef, state, 1);
  const auto after_dofs = bcs.allEssentialDofs();

  // Make sure that attempting to add a boundary condition
  // on already-used elements doesn't change the dofs
  EXPECT_EQ(before_dofs.Size(), after_dofs.Size());
  MPI_Barrier(MPI_COMM_WORLD);
}

enum TestTag
{
  Tag1 = 0,
  Tag2 = 1
};

enum OtherTag
{
  Fake1 = 0,
  Fake2 = 1
};

TEST(boundary_cond, filter_generics)
{
  MPI_Barrier(MPI_COMM_WORLD);
  constexpr int N    = 15;
  auto          mesh = mfem::Mesh::MakeCartesian2D(N, N, mfem::Element::TRIANGLE);
  mfem::ParMesh par_mesh(MPI_COMM_WORLD, mesh);

  BoundaryConditionManager bcs(par_mesh);
  auto                     coef = std::make_shared<mfem::ConstantCoefficient>(1);
  for (int i = 0; i < N; i++) {
    bcs.addGeneric({}, coef, TestTag::Tag1, 1);
    bcs.addGeneric({}, coef, TestTag::Tag2, 1);
  }

  int bcs_with_tag1 = 0;
  for (const auto& bc : bcs.genericsWithTag(TestTag::Tag1)) {
    EXPECT_TRUE(bc.tagEquals(TestTag::Tag1));
    // Also check that a different enum with the same underlying value will fail
    EXPECT_FALSE(bc.tagEquals(OtherTag::Fake1));
    bcs_with_tag1++;
  }
  EXPECT_EQ(bcs_with_tag1, N);

  int bcs_with_tag2 = 0;
  for (const auto& bc : bcs.genericsWithTag(TestTag::Tag2)) {
    EXPECT_TRUE(bc.tagEquals(TestTag::Tag2));
    EXPECT_FALSE(bc.tagEquals(OtherTag::Fake2));
    bcs_with_tag2++;
  }
  EXPECT_EQ(bcs_with_tag2, N);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(boundary_cond_helper, element_attribute_dof_list_scalar)
{
  MPI_Barrier(MPI_COMM_WORLD);

  auto mesh = mfem::Mesh::MakeCartesian3D(4, 4, 4, mfem::Element::HEXAHEDRON);
  mesh.SetAttribute(1, 2);
  mesh.SetAttribute(31, 2);

  mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
  int sdim = pmesh.SpaceDimension();

  mfem::VisItDataCollection visit_dc("attributeLook", &pmesh);
  visit_dc.Save();

  mfem::H1_FECollection h1_fec(1, sdim);
  mfem::ParFiniteElementSpace h1_fes(&pmesh, &h1_fec, 1);

  mfem::Array<int> elem_attr_is_ess(pmesh.attributes.Max());
  elem_attr_is_ess = 0;
  elem_attr_is_ess[2-1] = 1;
  mfem::Array<int> ess_tdof_list;
  serac::mfem_ext::GetEssentialTrueDofsFromElementAttribute(h1_fes, elem_attr_is_ess, ess_tdof_list, -1);

  EXPECT_EQ(ess_tdof_list.Size(), 16);
  EXPECT_EQ(ess_tdof_list[0], 25);
  EXPECT_EQ(ess_tdof_list[1], 26);
  EXPECT_EQ(ess_tdof_list[2], 30);
  EXPECT_EQ(ess_tdof_list[3], 31);
  EXPECT_EQ(ess_tdof_list[4], 50);
  EXPECT_EQ(ess_tdof_list[5], 51);
  EXPECT_EQ(ess_tdof_list[6], 55);
  EXPECT_EQ(ess_tdof_list[7], 56);
  EXPECT_EQ(ess_tdof_list[8], 81);
  EXPECT_EQ(ess_tdof_list[9], 82);
  EXPECT_EQ(ess_tdof_list[10], 86);
  EXPECT_EQ(ess_tdof_list[11], 87);
  EXPECT_EQ(ess_tdof_list[12], 106);
  EXPECT_EQ(ess_tdof_list[13], 107);
  EXPECT_EQ(ess_tdof_list[14], 111);
  EXPECT_EQ(ess_tdof_list[15], 112);
  
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(boundary_cond_helper, element_attribute_dof_list_vector)
{
  MPI_Barrier(MPI_COMM_WORLD);

  auto mesh = mfem::Mesh::MakeCartesian3D(4, 4, 4, mfem::Element::HEXAHEDRON);
  mesh.SetAttribute(2, 2);
  mesh.SetAttribute(3, 2);

  mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
  int sdim = pmesh.SpaceDimension();

  mfem::H1_FECollection h1_fec(1, sdim);
  mfem::ParFiniteElementSpace h1_fes(&pmesh, &h1_fec, 1);

  mfem::Array<int> elem_attr_is_ess(pmesh.attributes.Max());
  elem_attr_is_ess = 0;
  elem_attr_is_ess[2-1] = 1;
  mfem::Array<int> ess_tdof_list;
  serac::mfem_ext::GetEssentialTrueDofsFromElementAttribute(h1_fes, elem_attr_is_ess, ess_tdof_list, 1);

  EXPECT_EQ(ess_tdof_list.Size(), 12);
  EXPECT_EQ(ess_tdof_list[0], 1);
  EXPECT_EQ(ess_tdof_list[1], 2);
  EXPECT_EQ(ess_tdof_list[2], 6);
  EXPECT_EQ(ess_tdof_list[3], 7);
  EXPECT_EQ(ess_tdof_list[4], 26);
  EXPECT_EQ(ess_tdof_list[5], 27);
  EXPECT_EQ(ess_tdof_list[6], 31);
  EXPECT_EQ(ess_tdof_list[7], 32);
  EXPECT_EQ(ess_tdof_list[8], 51);
  EXPECT_EQ(ess_tdof_list[9], 52);
  EXPECT_EQ(ess_tdof_list[10], 56);
  EXPECT_EQ(ess_tdof_list[11], 57);

  MPI_Barrier(MPI_COMM_WORLD);
}

}  // namespace serac

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

// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/infrastructure/input.hpp"
#include "serac/coefficients/coefficient_extensions.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include <unordered_map>
#include <vector>
#include <iostream>
#include <algorithm>
#include "serac/infrastructure/set.hpp"
#include "serac/physics/utilities/finite_element_state.hpp"
#include "serac/numerics/mesh_utils.hpp"

class SlicErrorException : public std::exception {
};

class SetTest : public ::testing::Test {
protected:
  static void SetUpTestSuite()
  {
    axom::slic::setAbortFunction([]() { throw SlicErrorException{}; });
    axom::slic::setAbortOnError(true);
    axom::slic::setAbortOnWarning(false);
  }

  void SetUp() override {}
};

namespace serac {

TEST_F(SetTest, set)
{
  Set<int> a1(std::vector<int>{1, 2, 4, 2, 2, 3, 4});
  std::cout << "Set a1:" << std::endl << a1 << std::endl;
  EXPECT_EQ(a1, Set<int>({1, 2, 4, 2, 2, 3, 4}));

  Set<int> a2({1, 2, 3});
  std::cout << "Set a2:" << std::endl << a2 << std::endl;
  EXPECT_EQ(a2, Set<int>({1, 2, 3}));

  auto a3 = a1.getUnion(a2);
  std::cout << "Union of a1 and a2" << std::endl;
  std::cout << a3 << std::endl;
  // a1 = 1, 2, 4, 2, 2, 3, 4
  // a2 = 1, 2, 3
  // a3 = 1, 2, 4, 2, 2, 3, 4
  EXPECT_EQ(a3, Set<int>({1, 2, 4, 2, 2, 3, 4}));

  auto a4 = a1.getIntersection(a2);
  std::cout << "Intersection of a1 and a2" << std::endl;
  std::cout << a4 << std::endl;
  // a1 = 1, 2, 4, 2, 2, 3, 4
  // a2 = 1, 2, 3
  // a4 = 1, 2
  EXPECT_EQ(a4, Set<int>({1, 2}));

  auto a5 = a1.getDifference(a2);
  std::cout << "Difference of a1 and a2" << std::endl;
  std::cout << a5 << std::endl;
  // a1 = 1, 2, 4, 2, 2, 3, 4
  // a2 = 1, 2, 3
  // Note a5: is smaller than max_size(a1, a2)
  // a5 = , , , 2, 2, 3, 4
  EXPECT_EQ(a5.values({2, 3, 4}), (std::vector<decltype(a5)::index_type>{3, 4, 5, 6}));
  Set<int> check5({{2, {3, 4}}, {3, {5}}, {4, {6}}});
  EXPECT_EQ(a5, Set<int>({{2, {3, 4}}, {3, {5}}, {4, {6}}}));

  // Calculate the set operations on a set of attributes
  std::vector<Set<int>::index_type> specific_union = Union(a1.values({2}), a2.values({2, 3}));
  Set<int>                          a6({{2, specific_union}});
  std::cout << "Specific union" << std::endl;
  std::cout << a6 << std::endl;
  // a1.values({2})   = 1 3 4
  // a2.values({2,3}) = 1 2
  // union => 1 2 3 4
  EXPECT_EQ(a6.values(), (std::vector<decltype(a6)::index_type>{1, 2, 3, 4}));

  auto a7 = a1.getComplement({1, 3});
  std::cout << "Subset of a1 without (1,3)" << std::endl;
  std::cout << a7 << std::endl;
  EXPECT_EQ(a7.values(), (std::vector<decltype(a7)::index_type>{1, 2, 3, 4, 6}));
}

TEST_F(SetTest, flag_mesh)
{
  MPI_Barrier(MPI_COMM_WORLD);
  // Let's draw a flag
  double flag_width  = 1.5;
  double flag_height = 1.;
  auto   pmesh       = mesh::refineAndDistribute(buildRectangleMesh(100, 100, flag_width, flag_height));

  // first we need our 13 stripes
  // red = 1, white = 2
  auto stripes_coef = mfem::FunctionCoefficient([=](const mfem::Vector& coords) {
    auto stripe_height = flag_height / 13;
    auto stripe        = static_cast<int>(coords[1] / stripe_height);
    return stripe % 2 ? 2. : 1.;
  });

  // blue portion
  // blue = 3, white 2, outside = 1
  auto stars_coef = mfem::FunctionCoefficient([=](const mfem::Vector& coords) {
    auto   blue_height = flag_height / 2;
    auto   blue_width  = flag_width / 2.5;
    double color       = 1.;  // not blue
    if (coords[0] <= blue_width && coords[1] >= blue_height) {
      // could be blue, check to see if it's in a "star"
      // 9 down 11 across
      auto checker_height = blue_height / 9;
      auto checker_width  = blue_width / 11;
      auto checker_x      = static_cast<int>(coords[0] / checker_width);
      auto checker_y      = static_cast<int>((coords[1] - blue_height) / checker_height);
      if ((checker_x + checker_y * 11) % 2 == 0) {
        color = 2;
      } else {
        color = 3;
      }
    }
    return color;
  });

  mfem::VisItDataCollection visit(pmesh->GetComm(), "flag", pmesh.get());

  serac::FiniteElementState stripes(
      *pmesh, FiniteElementState::Options{.order      = 0,
                                          .vector_dim = 1,
                                          .coll = {std::make_unique<mfem::L2_FECollection>(0, pmesh->SpaceDimension())},
                                          .ordering = mfem::Ordering::byVDIM,
                                          .name     = "stripes",
                                          .alloc_gf = true});
  stripes.project(stripes_coef);

  auto stripes_attr_set =
      Set(mfem_ext::MakeAttributeList<std::vector<int>>(*pmesh, stripes_coef, mfem_ext::digitize::floor));

  serac::FiniteElementState stars(
      *pmesh, FiniteElementState::Options{.order      = 0,
                                          .vector_dim = 1,
                                          .coll = {std::make_unique<mfem::L2_FECollection>(0, pmesh->SpaceDimension())},
                                          .ordering = mfem::Ordering::byNODES,
                                          .name     = "stars",
                                          .alloc_gf = true});
  stars.project(stars_coef);

  auto stars_attr_set =
      Set(mfem_ext::MakeAttributeList<std::vector<int>>(*pmesh, stars_coef, mfem_ext::digitize::floor));

  // Set two additional bdr attributes: 1 = everything, 2 = bottom, 3 = right side
  auto bdr_assign_coef = mfem::FunctionCoefficient([&](const mfem::Vector& x) {
    if (fabs(x[1] - 0.) < 1.e-5) {
      return 2.;
    } else if (fabs(x[0] - flag_width) < 1.e-5) {
      return 3.;
    }
    return 1.;
  });

  mfem_ext::AssignMeshBdrAttributes(
      *pmesh, mfem_ext::MakeBdrAttributeList<std::vector<int>>(*pmesh, bdr_assign_coef, mfem_ext::digitize::floor));

  // we'll get the top blue corner
  auto blue = stars_attr_set.getComplement({1});

  // we want all the indices that composite the stripes
  // Take the difference of stripes_attr_set - (stars_attr_set = 1)
  auto red_white = stripes_attr_set.getDifference(blue);

  // Recombine the stripes sans top corner with the top corner
  auto flag = blue.getUnion(red_white);
  // Now red = 1, white = 2, blue = 3
  mfem_ext::AssignMeshElementAttributes(*pmesh, flag.toList());

  visit.RegisterField("stripes", &stripes.gridFunc());
  visit.RegisterField("stars", &stars.gridFunc());

  visit.SetCycle(0);
  visit.Save();

  // Integration tests on boundaries
  serac::FiniteElementState density(*pmesh, FiniteElementState::Options{.order      = 1,
                                                                        .vector_dim = 1,
                                                                        .coll       = {},
                                                                        .ordering   = mfem::Ordering::byVDIM,
                                                                        .name       = "density",
                                                                        .alloc_gf   = true});

  // First define PWConstantCoefficient that corresponds to red=1, white = 2, blue = 3
  mfem::Vector elem_densities(3);
  elem_densities[0] = 1.;
  elem_densities[1] = 0.0;
  elem_densities[2] = 0.0;
  auto density_coef = mfem::PWConstCoefficient(elem_densities);

  // integrate on the boundary of the bottom
  mfem::ParFiniteElementSpace& fes = *density.gridFunc().ParFESpace();
  mfem::ParGridFunction        one_gf(&fes);
  one_gf = 1.;

  mfem::Array<int> bottom_marker(3);
  bottom_marker    = 0;
  bottom_marker[1] = 1;

  auto surface_density_coef = mfem_ext::SurfaceElementAttrCoefficient(*pmesh, density_coef);

  /*
    Without SurfaceElementAttrCoefficient, the following integration routine will
    look at the bottom_marker, which tells mfem to integrate on attr = 2 on the boundary.
    However, density_coef will evaluate the bdrElementTransformation(attr = 2) which evaluates
    elem_densities[1], which is 0. This is not what we would expect in integrating the density along a boundary.

    So SurfaceElementAttrCoefficient allows you to specify attr = 2 bdr element integration.
    However, instead of passing the bdrElementTransformation(attr = 2), it passes the
    ElementTransformation associated with bdrElementTransformation(attr = 2) which is elem attr=1 "red".
    Thus, density_coef will evaluate as elem_attr=1 and yield the expected answer.
  */

  mfem::ParLinearForm bottom_wrong(&fes);
  bottom_wrong.AddBoundaryIntegrator(new mfem::BoundaryLFIntegrator(density_coef), bottom_marker);
  bottom_wrong.Assemble();
  double integrate_bottom_wrong = bottom_wrong(one_gf);
  EXPECT_EQ(0, integrate_bottom_wrong);

  mfem::ParLinearForm bottom(&fes);
  bottom.AddBoundaryIntegrator(new mfem::BoundaryLFIntegrator(surface_density_coef), bottom_marker);
  bottom.Assemble();
  double integrate_bottom = bottom(one_gf);
  EXPECT_NEAR(1.5, integrate_bottom, 5.e-2);

  mfem::Array<int> right_marker(3);
  right_marker    = 0;
  right_marker[2] = 1;

  mfem::ParLinearForm right(&fes);
  right.AddBoundaryIntegrator(new mfem::BoundaryLFIntegrator(surface_density_coef), right_marker);
  right.Assemble();
  double integrate_right = right(one_gf);
  EXPECT_NEAR(6. / 11., integrate_right, 5.e-2);

  // Reverse attributes: white = 1, red = 2, blue = 3 and integrate on right side
  std::vector<int> orig_elem_attr = flag.toList();
  std::vector<int> red_white_swapped(orig_elem_attr.size());
  std::transform(orig_elem_attr.begin(), orig_elem_attr.end(), red_white_swapped.begin(), [](int c) -> int {
    switch (c) {
      case 1:
        return 2;
      case 2:
        return 1;
      default:
        return 3;
    }
  });

  mfem::ParLinearForm right_swap(&fes);
  auto                modified_density_coef = mfem_ext::AttributeModifierCoefficient(red_white_swapped, density_coef);
  auto modified_surface_density_coef        = mfem_ext::SurfaceElementAttrCoefficient(*pmesh, modified_density_coef);
  right_swap.AddBoundaryIntegrator(new mfem::BoundaryLFIntegrator(modified_surface_density_coef), right_marker);
  right_swap.Assemble();
  double integrate_right_swap = right_swap(one_gf);
  EXPECT_NEAR(5. / 11., integrate_right_swap, 5.e-2);

  // Let's reassemble the original integration to verify that element attributes were hot-swapped and not changed
  // permanently
  right.Assemble();
  EXPECT_NEAR(6. / 11., right(one_gf), 5.e-2);

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

  axom::slic::SimpleLogger logger;  // create & initialize test logger, finalized when
                                    // exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}

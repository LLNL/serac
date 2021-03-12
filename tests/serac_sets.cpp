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

  void SetUp() override
  {
  }

};

namespace serac {


TEST_F(SetTest, set)
{
    Set<int> a1(std::vector<int>{1,2,4,2,2,3,4});
    Set<int> a2({1,2,3});
    std::cout << "values size: " << a1.keys_size() << std::endl;
    auto a3 = a1.getUnion(a2);
    std::cout << "Union of a1 and a2" << std::endl;
    std::cout << a3 << std::endl;
 
    auto a4 = a1.getIntersection(a2);
    std::cout << "Intersection of a1 and a2" << std::endl;
    std::cout << a4 << std::endl;
 
    auto a5 = a1.getDifference(a2);
    std::cout << "Difference of a1 and a2" << std::endl;
    std::cout << a5 << std::endl;
 
    // Just calculate the set operations on a set of attributes
    std::vector<Set<int>::index_type> specific_union = Union(a1.values({2}), a2.values({2,3}));
    Set<int> a6({{2, specific_union}});
    std::cout << "Specific union" << std::endl;
    std::cout << a6 << std::endl;
 
    auto a7 = a1.getComplement({1,3});
    std::cout << "Subset of a1 without (1,3)" << std::endl;
    std::cout << a7 << std::endl;
}

  TEST_F(SetTest, mesh)
  {

    MPI_Barrier(MPI_COMM_WORLD);
    // Let's draw a flag
    double flag_width = 1.5;
    double flag_height = 1.;
    auto pmesh = mesh::refineAndDistribute(buildRectangleMesh(100, 100, flag_width, flag_height));

    // first we need our 13 stripes
    // stripe_coef evaluates to 1 if red, 2 if white
    auto stripes_coef = mfem::FunctionCoefficient([=](const mfem::Vector & coords){
	auto stripe_height = flag_height/13;
	auto stripe = static_cast<int>(coords[1] / stripe_height);
	return stripe % 2 ? 1. : 2.;
      });

    // blue portion
    // blue = 3, white 2, outside = 1
    auto stars_coef = mfem::FunctionCoefficient([=](const mfem::Vector & coords) {
	auto blue_height = flag_height/2;
	auto blue_width = flag_width/2.5;
	double color = 1.; // not blue
	if (coords[0] <= blue_width && coords[1] >= blue_height) {
	  // could be blue, check to see if it's in a "star"
	  // 9 down 11 across
	  auto checker_height = blue_height/9;
	  auto checker_width = blue_width/11;
	  auto checker_x = static_cast<int>(coords[0]/checker_width);
	  auto checker_y = static_cast<int>((coords[1]-blue_height)/checker_height);
	  if ((checker_x + checker_y * 11) % 2 == 0) {
	    color = 3;
	  } else {
	    color = 2;
	  }
	}
	return color;
      });

    mfem::VisItDataCollection visit(pmesh->GetComm(), "flag", pmesh.get());

    // serac::FiniteElementState stripes(*pmesh, FiniteElementState::Options{
    // 	.order=0,.vector_dim=1,.coll={std::make_unique<mfem::L2_FECollection>( 0, pmesh->SpaceDimension())},.ordering=mfem::Ordering::byVDIM,.name="stripes"});
    auto fec = mfem::L2_FECollection(0, pmesh->SpaceDimension());
    auto pfes = mfem::ParFiniteElementSpace(pmesh.get(), &fec);
    auto stripes = mfem::ParGridFunction(&pfes);
    //    stripes.project(stripes_coef);
    stripes.ProjectCoefficient(stripes_coef);
    
    auto stripes_attr_set = Set(mfem_ext::MakeAttributeList<std::vector<int>>(*pmesh, stripes_coef, mfem_ext::digitize::floor));
    
    // serac::FiniteElementState stars(*pmesh, FiniteElementState::Options{
    // 	.order=0,.vector_dim=1,.coll={std::make_unique<mfem::L2_FECollection>( 0, pmesh->SpaceDimension())},.ordering=mfem::Ordering::byVDIM,.name="stars"});   
    //    stars.project(stars_coef);
    auto stars = mfem::ParGridFunction(&pfes);
    stars.ProjectCoefficient(stars_coef);
    
    auto stars_attr_set = Set(mfem_ext::MakeAttributeList<std::vector<int>>(*pmesh, stars_coef, mfem_ext::digitize::floor));

    // we'll get the top blue corner
    auto blue = stars_attr_set.getComplement({1});
    
    // we want all the indices that composite the stripes
    // intersect stars_attr_set = 1 and stripes_attr_set
    // auto red_white = Intersection(stars_attr_set.values({1}), stripe_attr_set.values({1,2}));
    auto red_white = stripes_attr_set.getDifference(blue);

    auto flag = blue.getUnion(red_white);
    auto flag_list = flag.toList();
    // mfem_ext::AssignMeshElementAttributes(*pmesh, flag_list);

    visit.RegisterField("stripes", &stripes);
    visit.RegisterField("stars", &stars);
    visit.SetCycle(0);
    visit.Save();

    
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

// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include <exception>

#include "numerics/mesh_utils.hpp"
#include "physics/utilities/boundary_condition.hpp"
#include "physics/utilities/equation_solver.hpp"

class SlicErrorException : public std::exception {
};

namespace serac {

TEST(serac_error_handling, equationsolver_bad_lin_solver)
{
  IterativeSolverParameters params;
  // Try a definitely wrong number to ensure that an invalid linear solver is detected
  params.lin_solver = static_cast<LinearSolver>(-7);
  EXPECT_THROW(EquationSolver(MPI_COMM_WORLD, params), SlicErrorException);
}

// Only need to test this when AmgX is **not** available
#ifndef MFEM_USE_AMGX
TEST(serac_error_handling, equationsolver_amgx_not_available)
{
  IterativeSolverParameters params;
  params.prec = AMGXPrec{};
  EXPECT_THROW(EquationSolver(MPI_COMM_WORLD, params), SlicErrorException);
}
#endif

TEST(serac_error_handling, bc_project_requires_state)
{
  auto mesh      = buildDiskMesh(10);
  int  num_attrs = mesh->bdr_attributes.Max();

  auto              coef = std::make_shared<mfem::ConstantCoefficient>();
  BoundaryCondition bc(coef, 0, std::set<int>{1}, num_attrs);
  EXPECT_THROW(bc.project(), SlicErrorException);

  FiniteElementState state(*mesh);
  bc.setTrueDofs(state);
  EXPECT_NO_THROW(bc.project());
}

TEST(serac_error_handling, bc_all_component_scalar_coef)
{
  auto coef = std::make_shared<mfem::ConstantCoefficient>();
  EXPECT_THROW(BoundaryCondition(coef, -1, std::set<int>{1}), SlicErrorException);
}

TEST(serac_error_handling, bc_one_component_vector_coef)
{
  mfem::Vector vec;
  auto         coef = std::make_shared<mfem::VectorConstantCoefficient>(vec);
  EXPECT_THROW(BoundaryCondition(coef, 0, std::set<int>{1}), SlicErrorException);
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

  UnitTestLogger logger;  // create & initialize test logger, finalized when
                          // exiting main scope
  axom::slic::setAbortFunction([]() { throw SlicErrorException{}; });
  axom::slic::setAbortOnError(true);
  axom::slic::setAbortOnWarning(false);

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}

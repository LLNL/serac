// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <exception>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>

#include "serac/infrastructure/cli.hpp"
#include "serac/infrastructure/initialize.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/thermal_conduction.hpp"
#include "serac/physics/boundary_conditions/boundary_condition.hpp"
#include "serac/numerics/equation_solver.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/serac_config.hpp"

class SlicErrorException : public std::exception {
};

namespace serac {

using mfem_ext::EquationSolver;

TEST(SeracErrorHandling, EquationSolverBadLinSolver)
{
  IterativeSolverOptions options;
  // Try a definitely wrong number to ensure that an invalid linear solver is detected
  options.lin_solver = static_cast<LinearSolver>(-7);
  EXPECT_THROW(EquationSolver(MPI_COMM_WORLD, options), SlicErrorException);
}

// Only need to test this when AmgX is **not** available
#ifndef MFEM_USE_AMGX
TEST(SeracErrorHandling, EquationSolverAmgxNotAvailable)
{
  IterativeSolverOptions options;
  options.prec = AMGXPrec{};
  EXPECT_THROW(EquationSolver(MPI_COMM_WORLD, options), SlicErrorException);
}
#endif

// Only need to test this when KINSOL is **not** available
#ifndef MFEM_USE_SUNDIALS
TEST(SeracErrorHandling, EquationSolverKinsolNotAvailable)
{
  auto lin_options             = ThermalConduction::defaultLinearOptions();
  auto nonlin_options          = ThermalConduction::defaultNonlinearOptions();
  nonlin_options.nonlin_solver = NonlinearSolver::KINFullStep;
  EXPECT_THROW(EquationSolver(MPI_COMM_WORLD, lin_options, nonlin_options), SlicErrorException);
}
#endif

TEST(SeracErrorHandling, BcProjectRequiresState)
{
  auto mesh      = mesh::refineAndDistribute(buildDiskMesh(10));
  int  num_attrs = mesh->bdr_attributes.Max();

  auto              coef = std::make_shared<mfem::ConstantCoefficient>();
  BoundaryCondition bc(coef, 0, std::set<int>{1}, num_attrs);
  EXPECT_THROW(bc.project(), SlicErrorException);
}

TEST(SeracErrorHandling, BcOneComponentVectorCoef)
{
  mfem::Vector vec;
  auto         coef = std::make_shared<mfem::VectorConstantCoefficient>(vec);
  EXPECT_THROW(BoundaryCondition(coef, 0, std::set<int>{1}), SlicErrorException);
}

TEST(SeracErrorHandling, BcOneComponentVectorCoefDofs)
{
  mfem::Vector     vec;
  auto             coef = std::make_shared<mfem::VectorConstantCoefficient>(vec);
  mfem::Array<int> dofs(1);
  dofs[0] = 1;
  EXPECT_THROW(BoundaryCondition(coef, 0, dofs), SlicErrorException);
}

TEST(SeracErrorHandling, BcProjectNoState)
{
  auto              coef = std::make_shared<mfem::ConstantCoefficient>(1.0);
  BoundaryCondition bc(coef, -1, std::set<int>{});
  EXPECT_THROW(bc.projectBdr(0.0), SlicErrorException);
}

TEST(SeracErrorHandling, BcRetrieveScalarCoef)
{
  auto              coef = std::make_shared<mfem::ConstantCoefficient>(1.0);
  BoundaryCondition bc(coef, -1, std::set<int>{});
  EXPECT_NO_THROW(bc.scalarCoefficient());
  EXPECT_THROW(bc.vectorCoefficient(), SlicErrorException);

  const auto& const_bc = bc;
  EXPECT_NO_THROW(const_bc.scalarCoefficient());
  EXPECT_THROW(const_bc.vectorCoefficient(), SlicErrorException);
}

TEST(SeracErrorHandling, BcRetrieveVecCoef)
{
  mfem::Vector      vec;
  auto              coef = std::make_shared<mfem::VectorConstantCoefficient>(vec);
  BoundaryCondition bc(coef, {}, std::set<int>{});
  EXPECT_NO_THROW(bc.vectorCoefficient());
  EXPECT_THROW(bc.scalarCoefficient(), SlicErrorException);

  const auto& const_bc = bc;
  EXPECT_NO_THROW(const_bc.vectorCoefficient());
  EXPECT_THROW(const_bc.scalarCoefficient(), SlicErrorException);
}

TEST(SeracErrorHandling, InvalidCmdlineArg)
{
  // The command is actually --input-file
  char const* fake_argv[] = {"serac", "--file", "input.lua"};
  const int   fake_argc   = 3;
  EXPECT_THROW(cli::defineAndParse(fake_argc, const_cast<char**>(fake_argv), ""), SlicErrorException);
}

TEST(SeracErrorHandling, NonexistentMeshPath)
{
  std::string mesh_path       = "nonexistent.mesh";
  std::string input_file_path = std::string(SERAC_REPO_DIR) + "/data/input_files/default.lua";
  EXPECT_THROW(input::findMeshFilePath(mesh_path, input_file_path), SlicErrorException);
}

}  // namespace serac

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  axom::slic::setAbortFunction([]() { throw SlicErrorException{}; });
  axom::slic::setAbortOnError(true);
  axom::slic::setAbortOnWarning(false);

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}

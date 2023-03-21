// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
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
#include "serac/physics/thermal_conduction_legacy.hpp"
#include "serac/physics/boundary_conditions/boundary_condition.hpp"
#include "serac/numerics/equation_solver.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/serac_config.hpp"

class SlicErrorException : public std::exception {
};

namespace serac {

using mfem_ext::EquationSolver;

TEST(ErrorHandling, EquationSolverBadLinSolver)
{
  IterativeSolverOptions options;
  // Try a definitely wrong number to ensure that an invalid linear solver is detected
  options.lin_solver = static_cast<LinearSolver>(-7);
  EXPECT_THROW(EquationSolver(MPI_COMM_WORLD, options), SlicErrorException);
}

// Only need to test this when AmgX is **not** available
#ifndef MFEM_USE_AMGX
TEST(ErrorHandling, EquationSolverAmgxNotAvailable)
{
  IterativeSolverOptions options;
  options.prec = AMGXPrec{};
  EXPECT_THROW(EquationSolver(MPI_COMM_WORLD, options), SlicErrorException);
}
#endif

// Only need to test this when KINSOL is **not** available
#ifndef MFEM_USE_SUNDIALS
TEST(ErrorHandling, EquationSolverKinsolNotAvailable)
{
  auto lin_options             = ThermalConductionLegacy::defaultLinearOptions();
  auto nonlin_options          = ThermalConductionLegacy::defaultNonlinearOptions();
  nonlin_options.nonlin_solver = NonlinearSolver::KINFullStep;
  EXPECT_THROW(EquationSolver(MPI_COMM_WORLD, lin_options, nonlin_options), SlicErrorException);
}
#endif

TEST(ErrorHandling, BcOneComponentVectorCoef)
{
  mfem::Vector vec;
  auto         coef = std::make_shared<mfem::VectorConstantCoefficient>(vec);

  auto mesh = mesh::refineAndDistribute(buildDiskMesh(10), 0, 0);

  mfem::H1_FECollection       coll(1, mesh->Dimension());
  mfem::ParFiniteElementSpace space(mesh.get(), &coll);

  EXPECT_THROW(BoundaryCondition(coef, 0, space, std::set<int>{1}), SlicErrorException);
}

TEST(ErrorHandling, BcOneComponentVectorCoefDofs)
{
  mfem::Vector     vec;
  auto             coef = std::make_shared<mfem::VectorConstantCoefficient>(vec);
  mfem::Array<int> dofs(1);
  dofs[0] = 1;

  auto mesh = mesh::refineAndDistribute(buildDiskMesh(10), 0, 0);

  mfem::H1_FECollection       coll(1, mesh->Dimension());
  mfem::ParFiniteElementSpace space(mesh.get(), &coll);

  EXPECT_THROW(BoundaryCondition(coef, 0, space, dofs), SlicErrorException);
}

TEST(ErrorHandling, BcRetrieveScalarCoef)
{
  auto mesh = mesh::refineAndDistribute(buildDiskMesh(10), 0, 0);

  mfem::H1_FECollection       coll(1, mesh->Dimension());
  mfem::ParFiniteElementSpace space(mesh.get(), &coll);

  auto              coef = std::make_shared<mfem::ConstantCoefficient>(1.0);
  BoundaryCondition bc(coef, -1, space, std::set<int>{});
  EXPECT_NO_THROW(bc.scalarCoefficient());
  EXPECT_THROW(bc.vectorCoefficient(), SlicErrorException);

  const auto& const_bc = bc;
  EXPECT_NO_THROW(const_bc.scalarCoefficient());
  EXPECT_THROW(const_bc.vectorCoefficient(), SlicErrorException);
}

TEST(ErrorHandling, BcRetrieveVecCoef)
{
  auto mesh = mesh::refineAndDistribute(buildDiskMesh(10), 0, 0);

  mfem::H1_FECollection       coll(1, mesh->Dimension());
  mfem::ParFiniteElementSpace space(mesh.get(), &coll);

  mfem::Vector      vec;
  auto              coef = std::make_shared<mfem::VectorConstantCoefficient>(vec);
  BoundaryCondition bc(coef, {}, space, std::set<int>{});
  EXPECT_NO_THROW(bc.vectorCoefficient());
  EXPECT_THROW(bc.scalarCoefficient(), SlicErrorException);

  const auto& const_bc = bc;
  EXPECT_NO_THROW(const_bc.vectorCoefficient());
  EXPECT_THROW(const_bc.scalarCoefficient(), SlicErrorException);
}

TEST(ErrorHandling, InvalidCmdlineArg)
{
  // The command is actually --input-file
  char const* fake_argv[] = {"serac", "--file", "input.lua"};
  const int   fake_argc   = 3;
  EXPECT_THROW(cli::defineAndParse(fake_argc, const_cast<char**>(fake_argv), ""), SlicErrorException);
}

TEST(ErrorHandling, NonexistentMeshPath)
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

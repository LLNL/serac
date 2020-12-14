// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include <exception>

#include "serac/infrastructure/cli.hpp"
#include "serac/infrastructure/initialize.hpp"
#include "serac/numerics/mesh_utils.hpp"
#include "serac/physics/thermal_conduction.hpp"
#include "serac/physics/utilities/boundary_condition.hpp"
#include "serac/physics/utilities/equation_solver.hpp"
#include "serac/serac_config.hpp"

class SlicErrorException : public std::exception {
};

namespace serac {

using mfem_ext::EquationSolver;

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

// Only need to test this when KINSOL is **not** available
#ifndef MFEM_USE_SUNDIALS
TEST(serac_error_handling, equationsolver_kinsol_not_available)
{
  auto lin_params             = ThermalConduction::defaultLinearParameters();
  auto nonlin_params          = ThermalConduction::defaultNonlinearParameters();
  nonlin_params.nonlin_solver = NonlinearSolver::KINFullStep;
  EXPECT_THROW(EquationSolver(MPI_COMM_WORLD, lin_params, nonlin_params), SlicErrorException);
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

TEST(serac_error_handling, bc_one_component_vector_coef)
{
  mfem::Vector vec;
  auto         coef = std::make_shared<mfem::VectorConstantCoefficient>(vec);
  EXPECT_THROW(BoundaryCondition(coef, 0, std::set<int>{1}), SlicErrorException);
}

TEST(serac_error_handling, bc_one_component_vector_coef_dofs)
{
  mfem::Vector     vec;
  auto             coef = std::make_shared<mfem::VectorConstantCoefficient>(vec);
  mfem::Array<int> dofs(1);
  dofs[0] = 1;
  EXPECT_THROW(BoundaryCondition(coef, 0, dofs), SlicErrorException);
}

TEST(serac_error_handling, bc_project_no_state)
{
  auto              coef = std::make_shared<mfem::ConstantCoefficient>(1.0);
  BoundaryCondition bc(coef, -1, std::set<int>{});
  EXPECT_THROW(bc.projectBdr(0.0), SlicErrorException);
}

TEST(serac_error_handling, bc_retrieve_scalar_coef)
{
  auto              coef = std::make_shared<mfem::ConstantCoefficient>(1.0);
  BoundaryCondition bc(coef, -1, std::set<int>{});
  EXPECT_NO_THROW(bc.scalarCoefficient());
  EXPECT_THROW(bc.vectorCoefficient(), SlicErrorException);

  const auto& const_bc = bc;
  EXPECT_NO_THROW(const_bc.scalarCoefficient());
  EXPECT_THROW(const_bc.vectorCoefficient(), SlicErrorException);
}

TEST(serac_error_handling, bc_retrieve_vec_coef)
{
  mfem::Vector      vec;
  auto              coef = std::make_shared<mfem::VectorConstantCoefficient>(vec);
  BoundaryCondition bc(coef, -1, std::set<int>{});
  EXPECT_NO_THROW(bc.vectorCoefficient());
  EXPECT_THROW(bc.scalarCoefficient(), SlicErrorException);

  const auto& const_bc = bc;
  EXPECT_NO_THROW(const_bc.vectorCoefficient());
  EXPECT_THROW(const_bc.scalarCoefficient(), SlicErrorException);
}

TEST(serac_error_handling, invalid_output_type)
{
  ThermalConduction physics(1, buildDiskMesh(100), ThermalConduction::defaultQuasistaticParameters());
  // Try a definitely wrong number to ensure that an invalid output type is detected
  EXPECT_THROW(physics.initializeOutput(static_cast<OutputType>(-7), ""), SlicErrorException);
}

TEST(serac_error_handling, invalid_cmdline_arg)
{
  // The command is actually --input-file
  char const* fake_argv[] = {"serac", "--file", "input.lua"};
  const int   fake_argc   = 3;
  int         rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  EXPECT_THROW(cli::defineAndParse(fake_argc, const_cast<char**>(fake_argv), rank, ""), SlicErrorException);
}

TEST(serac_error_handling, nonexistent_mesh_path)
{
  std::string mesh_path       = "nonexistent.mesh";
  std::string input_file_path = std::string(SERAC_REPO_DIR) + "/data/input_files/default.lua";
  EXPECT_THROW(input::findMeshFilePath(mesh_path, input_file_path), SlicErrorException);
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

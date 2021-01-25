// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid.hpp"

#include <fstream>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/coefficients/coefficient_extensions.hpp"
#include "serac/infrastructure/input.hpp"
#include "serac/numerics/mesh_utils.hpp"
#include "serac/serac_config.hpp"
#include "test_utilities.hpp"

namespace serac {

using test_utils::InputFileTest;

TEST_P(InputFileTest, nonlin_solid)
{
  MPI_Barrier(MPI_COMM_WORLD);
  std::string input_file_path =
      std::string(SERAC_REPO_DIR) + "/data/input_files/tests/nonlinear_solid/" + GetParam() + ".lua";
  test_utils::runModuleTest<Solid>(input_file_path);
  MPI_Barrier(MPI_COMM_WORLD);
}

const std::string input_files[] = {"dyn_solve",
                                   "dyn_direct_solve",
#ifdef MFEM_USE_SUNDIALS
                                   "dyn_linesearch_solve",
#endif
#ifdef MFEM_USE_AMGX
                                   "dyn_amgx_solve",
#endif
                                   "qs_solve",
                                   "qs_direct_solve"};

INSTANTIATE_TEST_SUITE_P(NonlinearSolidInputFileTests, InputFileTest, ::testing::ValuesIn(input_files));

TEST(nonlinear_solid_solver, qs_custom_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  std::string input_file_path = std::string(SERAC_REPO_DIR) + "/data/input_files/tests/nonlinear_solid/qs_solve.lua";

  // Create DataStore
  axom::sidre::DataStore datastore;

  // Initialize Inlet and read input file
  auto inlet = serac::input::initialize(datastore, input_file_path);

  test_utils::defineTestSchema<Solid>(inlet);

  // Build the mesh
  auto mesh_options   = inlet["main_mesh"].get<serac::mesh::InputOptions>();
  auto full_mesh_path = serac::input::findMeshFilePath(mesh_options.relative_mesh_file_name, input_file_path);
  auto mesh = serac::buildMeshFromFile(full_mesh_path, mesh_options.ser_ref_levels, mesh_options.par_ref_levels);

  // Define the solid solver object
  auto solid_solver_options = inlet["nonlinear_solid"].get<serac::Solid::InputOptions>();

  // Simulate a custom solver by manually building the linear solver and passing it in
  // The custom solver built here should be identical to what is internally built in the
  // qs_solve test
  auto custom_options = inlet["nonlinear_solid/stiffness_solver/linear"].get<serac::LinearSolverOptions>();
  auto iter_options   = std::get<serac::IterativeSolverOptions>(custom_options);
  auto custom_solver  = std::make_unique<mfem::MINRESSolver>(MPI_COMM_WORLD);
  custom_solver->SetRelTol(iter_options.rel_tol);
  custom_solver->SetAbsTol(iter_options.abs_tol);
  custom_solver->SetMaxIter(iter_options.max_iter);
  custom_solver->SetPrintLevel(iter_options.print_level);

  solid_solver_options.solver_options.H_lin_options = CustomSolverOptions{custom_solver.get()};
  Solid solid_solver(mesh, solid_solver_options);

  // Initialize the output
  solid_solver.initializeOutput(serac::OutputType::VisIt, "static_solid");

  // Complete the solver setup
  solid_solver.completeSetup();

  double dt = inlet["dt"];
  solid_solver.advanceTimestep(dt);

  solid_solver.outputState();

  int          dim = mesh->Dimension();
  mfem::Vector zero(dim);
  zero = 0.0;
  mfem::VectorConstantCoefficient zerovec(zero);

  double x_norm = solid_solver.displacement().gridFunc().ComputeLpError(2.0, zerovec);

  EXPECT_NEAR(inlet["expected_x_l2norm"], x_norm, inlet["epsilon"]);

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

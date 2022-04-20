// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid.hpp"

#include <fstream>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/physics/coefficients/coefficient_extensions.hpp"
#include "serac/infrastructure/input.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/serac_config.hpp"
#include "test_utilities.hpp"

namespace serac {

using test_utils::InputFileTest;

TEST_P(InputFileTest, solid)
{
  MPI_Barrier(MPI_COMM_WORLD);
  std::string input_file_path = std::string(SERAC_REPO_DIR) + "/data/input_files/tests/solid/" + GetParam() + ".lua";
  test_utils::runModuleTest<Solid>(input_file_path, GetParam());
  MPI_Barrier(MPI_COMM_WORLD);
}

const std::string input_files[] = {"dyn_solve",      "dyn_direct_solve",
// TODO Disabled while we diagnose the non-deterministic sundials error
/*
#ifdef MFEM_USE_SUNDIALS
                                   "dyn_linesearch_solve",
#endif
*/
#ifdef MFEM_USE_AMGX
                                   "dyn_amgx_solve",
#endif
                                   "qs_solve",       "qs_direct_solve",  "qs_linear"};

INSTANTIATE_TEST_SUITE_P(SolidInputFileTests, InputFileTest, ::testing::ValuesIn(input_files));

TEST(solid_solver, qs_custom_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  std::string input_file_path = std::string(SERAC_REPO_DIR) + "/data/input_files/tests/solid/qs_solve.lua";

  // Create DataStore
  axom::sidre::DataStore datastore;

  // Initialize Inlet and read input file
  auto inlet = serac::input::initialize(datastore, input_file_path);
  serac::StateManager::initialize(datastore, "solid_qs_custom_solve");

  test_utils::defineTestSchema<Solid>(inlet);

  // Build the mesh
  auto mesh_options = inlet["main_mesh"].get<serac::mesh::InputOptions>();
  if (const auto file_opts = std::get_if<serac::mesh::FileInputOptions>(&mesh_options.extra_options)) {
    file_opts->absolute_mesh_file_name =
        serac::input::findMeshFilePath(file_opts->relative_mesh_file_name, input_file_path);
  }
  auto mesh = serac::mesh::buildParallelMesh(mesh_options);
  serac::StateManager::setMesh(std::move(mesh));

  // Define the solid solver object
  auto solid_solver_options = inlet["solid"].get<serac::Solid::InputOptions>();

  // Simulate a custom solver by manually building the linear solver and passing it in
  // The custom solver built here should be identical to what is internally built in the
  // qs_solve test
  auto custom_options = inlet["solid/equation_solver/linear"].get<serac::LinearSolverOptions>();
  auto iter_options   = std::get<serac::IterativeSolverOptions>(custom_options);
  auto custom_solver  = std::make_unique<mfem::MINRESSolver>(MPI_COMM_WORLD);
  custom_solver->SetRelTol(iter_options.rel_tol);
  custom_solver->SetAbsTol(iter_options.abs_tol);
  custom_solver->SetMaxIter(iter_options.max_iter);
  custom_solver->SetPrintLevel(iter_options.print_level);

  solid_solver_options.solver_options.H_lin_options = CustomSolverOptions{custom_solver.get()};
  Solid solid_solver(solid_solver_options);

  // Initialize the output
  solid_solver.initializeOutput(serac::OutputType::VisIt, "static_solid");

  // Complete the solver setup
  solid_solver.completeSetup();

  double dt = inlet["dt"];
  solid_solver.advanceTimestep(dt);

  solid_solver.outputState();

  EXPECT_NEAR(inlet["expected_displacement_l2norm"], norm(solid_solver.displacement()), inlet["epsilon"]);

  // 0 = R(u) + K(u) du
  // u_sol = u + du
  // R(u_sol) < exit_tol
  // -R(u_sol) = K(u_sol) du_sol
  // R(u_sol + du_sol) < R(u_sol)

  auto         residual = solid_solver.currentResidual();
  mfem::Vector du(residual.Size());
  du = 0.0;

  mfem::MINRESSolver minres_solver(MPI_COMM_WORLD);
  minres_solver.SetOperator(solid_solver.currentGradient());
  minres_solver.Mult(residual, du);

  // modify the displacement just to recompute the residual
  solid_solver.displacement().trueVec() += du;
  auto residual_lower = solid_solver.currentResidual();
  EXPECT_LE(residual.Norml2(), residual_lower.Norml2());

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

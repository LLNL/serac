// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/nonlinear_solid.hpp"

#include <fstream>
#include <gtest/gtest.h>

#include "mfem.hpp"

#include "serac/coefficients/coefficient_extensions.hpp"
#include "serac/infrastructure/input.hpp"
#include "serac/numerics/mesh_utils.hpp"
#include "serac/serac_config.hpp"
#include "test_utilities.hpp"

namespace serac {

class InputFileTest : public ::testing::TestWithParam<std::string> {
};

TEST_P(InputFileTest, nonlin_solid)
{
  MPI_Barrier(MPI_COMM_WORLD);
  std::string input_file_path =
      std::string(SERAC_REPO_DIR) + "/data/input_files/tests/nonlinear_solid/" + GetParam() + ".lua";
  test_utils::runNonlinSolidTest(input_file_path);
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
                                   "qs_direct_solve",
                                   "qs_attribute_solve"};

INSTANTIATE_TEST_SUITE_P(NonlinearSolidInputFileTests, InputFileTest, ::testing::ValuesIn(input_files));

TEST(nonlinear_solid_solver, qs_custom_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  std::string input_file_path = std::string(SERAC_REPO_DIR) + "/data/input_files/tests/nonlinear_solid/qs_solve.lua";

  // Create DataStore
  axom::sidre::DataStore datastore;

  // Initialize Inlet and read input file
  auto inlet = serac::input::initialize(datastore, input_file_path);

  test_utils::defineNonlinSolidInputFileSchema(inlet);

  // Build the mesh
  auto mesh_info      = inlet["main_mesh"].get<serac::mesh::InputInfo>();
  auto full_mesh_path = serac::input::findMeshFilePath(mesh_info.relative_mesh_file_name, input_file_path);
  auto mesh           = serac::buildMeshFromFile(full_mesh_path, mesh_info.ser_ref_levels, mesh_info.par_ref_levels);

  // Define the solid solver object
  auto solid_solver_info = inlet["nonlinear_solid"].get<serac::NonlinearSolid::InputInfo>();

  // Simulate a custom solver by manually building the linear solver and passing it in
  // The custom solver built here should be identical to what is internally built in the
  // qs_solve test
  auto custom_params = inlet["nonlinear_solid/stiffness_solver/linear"].get<serac::LinearSolverParameters>();
  auto iter_params   = std::get<serac::IterativeSolverParameters>(custom_params);
  auto custom_solver = std::make_unique<mfem::MINRESSolver>(MPI_COMM_WORLD);
  custom_solver->SetRelTol(iter_params.rel_tol);
  custom_solver->SetAbsTol(iter_params.abs_tol);
  custom_solver->SetMaxIter(iter_params.max_iter);
  custom_solver->SetPrintLevel(iter_params.print_level);

  solid_solver_info.solver_params.H_lin_params = CustomSolverParameters{custom_solver.get()};
  NonlinearSolid solid_solver(mesh, solid_solver_info);

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

TEST(nonlinear_solid_solver, qs_component_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  std::string input_file_path =
      std::string(SERAC_REPO_DIR) + "/data/input_files/tests/nonlinear_solid/qs_component_solve.lua";

  // Create DataStore
  axom::sidre::DataStore datastore;

  // Initialize Inlet and read input file
  auto inlet = serac::input::initialize(datastore, input_file_path);

  test_utils::defineNonlinSolidInputFileSchema(inlet);

  // Build the mesh
  auto mesh_info      = inlet["main_mesh"].get<serac::mesh::InputInfo>();
  auto full_mesh_path = serac::input::findMeshFilePath(mesh_info.relative_mesh_file_name, input_file_path);
  auto mesh           = serac::buildMeshFromFile(full_mesh_path, mesh_info.ser_ref_levels, mesh_info.par_ref_levels);

  // Define the solid solver object
  auto                  solid_solver_info = inlet["nonlinear_solid"].get<serac::NonlinearSolid::InputInfo>();
  serac::NonlinearSolid solid_solver(mesh, solid_solver_info);

  int dim = mesh->Dimension();

  // define the displacement vector
  const auto& disp_bc   = solid_solver_info.boundary_conditions.at("displacement");
  auto        disp_coef = std::make_shared<mfem::FunctionCoefficient>(disp_bc.coef_info.scalar_func);

  // Create an indicator function to set all vertices that are x=0
  mfem::VectorFunctionCoefficient zero_bc(dim, [](const mfem::Vector& x, mfem::Vector& X) {
    X = 0.;
    for (int i = 0; i < X.Size(); i++)
      if (std::abs(x[i]) < 1.e-13) {
        X[i] = 1.;
      }
  });

  mfem::Array<int> ess_corner_bc_list = makeTrueEssList(solid_solver.displacement().space(), zero_bc);

  solid_solver.setTrueDofs(ess_corner_bc_list, disp_coef, disp_bc.coef_info.component);

  // Setup glvis output
  solid_solver.initializeOutput(serac::OutputType::GLVis, "component_bc");

  // Complete the solver setup
  solid_solver.completeSetup();

  double dt = inlet["dt"];
  solid_solver.advanceTimestep(dt);

  // Output the state
  solid_solver.outputState();

  auto state = solid_solver.getState();

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

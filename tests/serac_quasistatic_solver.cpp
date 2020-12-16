// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include <fstream>

#include "mfem.hpp"
#include "serac/coefficients/loading_functions.hpp"
#include "serac/infrastructure/input.hpp"
#include "serac/numerics/mesh_utils.hpp"
#include "serac/physics/nonlinear_solid.hpp"
#include "serac/serac_config.hpp"
#include "test_utilities.hpp"

namespace serac {

TEST(nonlinear_solid_solver, qs_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  std::string input_file_path =
      std::string(SERAC_REPO_DIR) + "/data/input_files/tests/nonlinear_solid/serac_quasistatic_solver/qs_solve.lua";

  // Create DataStore
  axom::sidre::DataStore datastore;

  // Initialize Inlet and read input file
  auto inlet = serac::input::initialize(datastore, input_file_path);

  testing::defineNonlinSolidInputFileSchema(inlet);

  // Build the mesh
  auto mesh_info      = inlet["main_mesh"].get<serac::mesh::InputOptions>();
  auto full_mesh_path = serac::input::findMeshFilePath(mesh_info.relative_mesh_file_name, input_file_path);
  auto mesh           = serac::buildMeshFromFile(full_mesh_path, mesh_info.ser_ref_levels, mesh_info.par_ref_levels);

  // Define the solid solver object
  auto           solid_solver_info = inlet["nonlinear_solid"].get<serac::NonlinearSolid::InputOptions>();
  NonlinearSolid solid_solver(mesh, solid_solver_info);

  int dim = mesh->Dimension();

  // define the displacement vector
  mfem::Vector disp(dim);
  disp = 0.0;

  auto disp_coef = std::make_shared<mfem::VectorConstantCoefficient>(disp);

  // define the traction vector
  auto traction = inlet["nonlinear_solid/traction"].get<mfem::Vector>();
  traction.SetSize(dim);  // The vector in the input file is 3D, so shrink it if needed
  auto traction_coef = std::make_shared<mfem::VectorConstantCoefficient>(traction);

  // Pass the BC information to the solver object
  for (const auto& bc : solid_solver_info.boundary_conditions) {
    if (bc.name == "displacement") {
      solid_solver.setDisplacementBCs(bc.attrs, disp_coef);
    } else if (bc.name == "traction") {
      solid_solver.setTractionBCs(bc.attrs, traction_coef);
    } else {
      SLIC_WARNING("Ignoring unrecognized boundary condition: " << bc.name);
    }
  }

  // Initialize the output
  solid_solver.initializeOutput(serac::OutputType::VisIt, "static_solid");

  // Complete the solver setup
  solid_solver.completeSetup();

  double dt = inlet["dt"];
  solid_solver.advanceTimestep(dt);

  solid_solver.outputState();

  mfem::Vector zero(dim);
  zero = 0.0;
  mfem::VectorConstantCoefficient zerovec(zero);

  double x_norm = solid_solver.displacement().gridFunc().ComputeLpError(2.0, zerovec);

  EXPECT_NEAR(inlet["expected_x_l2norm"], x_norm, inlet["epsilon"]);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(nonlinear_solid_solver, qs_direct_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  std::string input_file_path =
      std::string(SERAC_REPO_DIR) + "/data/input_files/tests/nonlinear_solid/serac_quasistatic_solver/qs_solve.lua";

  // Create DataStore
  axom::sidre::DataStore datastore;

  // Initialize Inlet and read input file
  auto inlet = serac::input::initialize(datastore, input_file_path);

  testing::defineNonlinSolidInputFileSchema(inlet);

  // Build the mesh
  auto mesh_info      = inlet["main_mesh"].get<serac::mesh::InputOptions>();
  auto full_mesh_path = serac::input::findMeshFilePath(mesh_info.relative_mesh_file_name, input_file_path);
  auto mesh           = serac::buildMeshFromFile(full_mesh_path, mesh_info.ser_ref_levels, mesh_info.par_ref_levels);

  // Define the solid solver object
  auto solid_solver_info = inlet["nonlinear_solid"].get<serac::NonlinearSolid::InputOptions>();
  // FIXME: These should be moved to part of the schema once the contains() logic is updated in Inlet
  solid_solver_info.solver_params.H_lin_params = DirectSolverOptions{0};
  NonlinearSolid solid_solver(mesh, solid_solver_info);

  int dim = mesh->Dimension();

  // define the displacement vector
  mfem::Vector disp(dim);
  disp = 0.0;

  auto disp_coef = std::make_shared<mfem::VectorConstantCoefficient>(disp);

  // define the traction vector
  auto traction = inlet["nonlinear_solid/traction"].get<mfem::Vector>();
  traction.SetSize(dim);  // The vector in the input file is 3D, so shrink it if needed
  auto traction_coef = std::make_shared<mfem::VectorConstantCoefficient>(traction);

  // Pass the BC information to the solver object
  for (const auto& bc : solid_solver_info.boundary_conditions) {
    if (bc.name == "displacement") {
      solid_solver.setDisplacementBCs(bc.attrs, disp_coef);
    } else if (bc.name == "traction") {
      solid_solver.setTractionBCs(bc.attrs, traction_coef);
    } else {
      SLIC_WARNING("Ignoring unrecognized boundary condition: " << bc.name);
    }
  }

  // Initialize the output
  solid_solver.initializeOutput(serac::OutputType::VisIt, "static_solid");

  // Complete the solver setup
  solid_solver.completeSetup();

  double dt = inlet["dt"];
  solid_solver.advanceTimestep(dt);

  solid_solver.outputState();

  mfem::Vector zero(dim);
  zero = 0.0;
  mfem::VectorConstantCoefficient zerovec(zero);

  double x_norm = solid_solver.displacement().gridFunc().ComputeLpError(2.0, zerovec);

  EXPECT_NEAR(inlet["expected_x_l2norm"], x_norm, inlet["epsilon"]);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(nonlinear_solid_solver, qs_custom_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  std::string input_file_path =
      std::string(SERAC_REPO_DIR) + "/data/input_files/tests/nonlinear_solid/serac_quasistatic_solver/qs_solve.lua";

  // Create DataStore
  axom::sidre::DataStore datastore;

  // Initialize Inlet and read input file
  auto inlet = serac::input::initialize(datastore, input_file_path);

  testing::defineNonlinSolidInputFileSchema(inlet);

  // Build the mesh
  auto mesh_info      = inlet["main_mesh"].get<serac::mesh::InputOptions>();
  auto full_mesh_path = serac::input::findMeshFilePath(mesh_info.relative_mesh_file_name, input_file_path);
  auto mesh           = serac::buildMeshFromFile(full_mesh_path, mesh_info.ser_ref_levels, mesh_info.par_ref_levels);

  // Define the solid solver object
  auto solid_solver_info = inlet["nonlinear_solid"].get<serac::NonlinearSolid::InputOptions>();

  // Simulate a custom solver by manually building the linear solver and passing it in
  // The custom solver built here should be identical to what is internally built in the
  // qs_solve test
  auto custom_params = inlet["nonlinear_solid/stiffness_solver/linear"].get<serac::IterativeSolverOptions>();
  auto custom_solver = std::make_unique<mfem::MINRESSolver>(MPI_COMM_WORLD);
  custom_solver->SetRelTol(custom_params.rel_tol);
  custom_solver->SetAbsTol(custom_params.abs_tol);
  custom_solver->SetMaxIter(custom_params.max_iter);
  custom_solver->SetPrintLevel(custom_params.print_level);

  solid_solver_info.solver_params.H_lin_params = CustomSolverOptions{custom_solver.get()};
  NonlinearSolid solid_solver(mesh, solid_solver_info);

  int dim = mesh->Dimension();

  // define the displacement vector
  mfem::Vector disp(dim);
  disp = 0.0;

  auto disp_coef = std::make_shared<mfem::VectorConstantCoefficient>(disp);

  // define the traction vector
  auto traction = inlet["nonlinear_solid/traction"].get<mfem::Vector>();
  traction.SetSize(dim);  // The vector in the input file is 3D, so shrink it if needed
  auto traction_coef = std::make_shared<mfem::VectorConstantCoefficient>(traction);

  // Pass the BC information to the solver object
  for (const auto& bc : solid_solver_info.boundary_conditions) {
    if (bc.name == "displacement") {
      solid_solver.setDisplacementBCs(bc.attrs, disp_coef);
    } else if (bc.name == "traction") {
      solid_solver.setTractionBCs(bc.attrs, traction_coef);
    } else {
      SLIC_WARNING("Ignoring unrecognized boundary condition: " << bc.name);
    }
  }

  // Initialize the output
  solid_solver.initializeOutput(serac::OutputType::VisIt, "static_solid");

  // Complete the solver setup
  solid_solver.completeSetup();

  double dt = inlet["dt"];
  solid_solver.advanceTimestep(dt);

  solid_solver.outputState();

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

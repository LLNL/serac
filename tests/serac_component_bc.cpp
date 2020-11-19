// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include <fstream>

#include "coefficients/coefficient_extensions.hpp"
#include "infrastructure/input.hpp"
#include "mfem.hpp"
#include "numerics/mesh_utils.hpp"
#include "physics/nonlinear_solid.hpp"
#include "serac_config.hpp"

namespace serac {

void defineInputFileSchema(axom::inlet::Inlet& inlet)
{
  // Simulation time parameters
  inlet.addDouble("dt", "Time step.");

  // Integration test parameters
  inlet.addDouble("expected_l2norm", "Correct L2 norm of the solution field");
  inlet.addDouble("epsilon", "Threshold to be used in the comparison");

  auto& mesh_table = inlet.addTable("main_mesh", "The main mesh for the problem");
  serac::mesh::InputInfo::defineInputFileSchema(mesh_table);

  // Physics
  auto& solid_solver_table = inlet.addTable("nonlinear_solid", "Finite deformation solid mechanics module");
  serac::NonlinearSolid::InputInfo::defineInputFileSchema(solid_solver_table);

  // Verify input file
  if (!inlet.verify()) {
    SLIC_ERROR("Input file failed to verify.");
  }
}

TEST(component_bc, qs_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  std::string input_file_path =
      std::string(SERAC_REPO_DIR) + "/data/input_files/tests/nonlinear_solid/serac_component_bc/qs_solve.lua";

  // Create DataStore
  axom::sidre::DataStore datastore;

  // Initialize Inlet and read input file
  auto inlet = serac::input::initialize(datastore, input_file_path);

  defineInputFileSchema(inlet);

  // Build the mesh
  auto mesh_info      = inlet["main_mesh"].get<serac::mesh::InputInfo>();
  auto full_mesh_path = serac::input::findMeshFilePath(mesh_info.relative_mesh_file_name, input_file_path);
  auto mesh           = serac::buildMeshFromFile(full_mesh_path, mesh_info.ser_ref_levels, mesh_info.par_ref_levels);

  // Define the solid solver object
  auto                  solid_solver_info = inlet["nonlinear_solid"].get<serac::NonlinearSolid::InputInfo>();
  serac::NonlinearSolid solid_solver(mesh, solid_solver_info);

  int dim = mesh->Dimension();

  // define the displacement vector
  auto disp_coef = std::make_shared<mfem::FunctionCoefficient>([](const mfem::Vector& x) { return x[0] * -1.0e-1; });

  // Pass the BC information to the solver object setting only the z direction
  for (const auto& bc : solid_solver_info.boundary_conditions) {
    if (bc.name == "displacement") {
      solid_solver.setDisplacementBCs(bc.attrs, disp_coef, 0);
    } else {
      SLIC_WARNING("Ignoring unrecognized boundary condition: " << bc.name);
    }
  }

  // Create an indicator function to set all vertices that are x=0
  mfem::VectorFunctionCoefficient zero_bc(dim, [](const mfem::Vector& x, mfem::Vector& X) {
    X = 0.;
    for (int i = 0; i < X.Size(); i++)
      if (std::abs(x[i]) < 1.e-13) {
        X[i] = 1.;
      }
  });

  mfem::Array<int> ess_corner_bc_list = makeTrueEssList(solid_solver.displacement().space(), zero_bc);

  solid_solver.setTrueDofs(ess_corner_bc_list, disp_coef, 0);

  // Setup glvis output
  solid_solver.initializeOutput(serac::OutputType::VisIt, "component_bc");

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

  EXPECT_NEAR(inlet["expected_l2norm"], x_norm, inlet["epsilon"]);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(component_bc, qs_attribute_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  std::string input_file_path =
      std::string(SERAC_REPO_DIR) + "/data/input_files/tests/nonlinear_solid/serac_component_bc/qs_attribute_solve.lua";

  // Create DataStore
  axom::sidre::DataStore datastore;

  // Initialize Inlet and read input file
  auto inlet = serac::input::initialize(datastore, input_file_path);

  defineInputFileSchema(inlet);

  // Build the mesh
  auto mesh_info      = inlet["main_mesh"].get<serac::mesh::InputInfo>();
  auto full_mesh_path = serac::input::findMeshFilePath(mesh_info.relative_mesh_file_name, input_file_path);
  auto mesh           = serac::buildMeshFromFile(full_mesh_path, mesh_info.ser_ref_levels, mesh_info.par_ref_levels);

  // Define the solid solver object
  auto                  solid_solver_info = inlet["nonlinear_solid"].get<serac::NonlinearSolid::InputInfo>();
  serac::NonlinearSolid solid_solver(mesh, solid_solver_info);

  int dim = mesh->Dimension();

  // define the displacement vector
  auto disp_x_coef = std::make_shared<mfem::FunctionCoefficient>([](const mfem::Vector& x) { return x[0] * 3.0e-2; });
  auto disp_y_coef = std::make_shared<mfem::FunctionCoefficient>([](const mfem::Vector& x) { return x[1] * -5.0e-2; });

  // Pass the BC information to the solver object setting only the z direction
  for (const auto& bc : solid_solver_info.boundary_conditions) {
    if (bc.name == "displacement_x") {
      solid_solver.setDisplacementBCs(bc.attrs, disp_x_coef, 0);
    } else if (bc.name == "displacement_y") {
      solid_solver.setDisplacementBCs(bc.attrs, disp_y_coef, 1);
    } else {
      SLIC_WARNING("Ignoring unrecognized boundary condition: " << bc.name);
    }
  }

  // Setup glvis output
  solid_solver.initializeOutput(serac::OutputType::GLVis, "component_attr_bc");

  // Complete the solver setup
  solid_solver.completeSetup();

  double dt = inlet["dt"];
  solid_solver.advanceTimestep(dt);

  // Output the state
  solid_solver.outputState();

  mfem::Vector zero(dim);
  zero = 0.0;
  mfem::VectorConstantCoefficient zerovec(zero);

  double x_norm = solid_solver.displacement().gridFunc().ComputeLpError(2.0, zerovec);

  EXPECT_NEAR(inlet["expected_l2norm"], x_norm, inlet["epsilon"]);

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

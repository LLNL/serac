// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
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
#include "serac/physics/utilities/state_manager.hpp"
#include "serac/serac_config.hpp"
#include "test_utilities.hpp"

namespace serac {

TEST(solid_solver, qs_attribute_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);
  std::string input_file_path = std::string(SERAC_REPO_DIR) + "/data/input_files/tests/solid/qs_attribute_solve.lua";
  test_utils::runModuleTest<Solid>(input_file_path, "qs_attribute_solve");
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(solid_solver, qs_component_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);
  std::string input_file_path = std::string(SERAC_REPO_DIR) + "/data/input_files/tests/solid/qs_component_solve.lua";

  // Create DataStore
  axom::sidre::DataStore datastore;

  // Initialize Inlet and read input file
  auto inlet = serac::input::initialize(datastore, input_file_path);
  serac::StateManager::initialize(datastore);

  test_utils::defineTestSchema<Solid>(inlet);

  // Build the mesh
  auto mesh_options = inlet["main_mesh"].get<serac::mesh::InputOptions>();
  if (const auto file_opts = std::get_if<serac::mesh::FileInputOptions>(&mesh_options.extra_options)) {
    file_opts->absolute_mesh_file_name =
        serac::input::findMeshFilePath(file_opts->relative_mesh_file_name, input_file_path);
  }
  auto      mesh = serac::mesh::buildParallelMesh(mesh_options);
  const int dim  = mesh->Dimension();
  serac::StateManager::setMesh(std::move(mesh));

  // Define the solid solver object
  auto         solid_solver_options = inlet["solid"].get<serac::Solid::InputOptions>();
  serac::Solid solid_solver(solid_solver_options);

  // define the displacement vector
  const auto&                        disp_bc = solid_solver_options.boundary_conditions.at("displacement");
  std::shared_ptr<mfem::Coefficient> disp_coef(disp_bc.coef_opts.constructScalar());

  // Create an indicator function to set all vertices that are x=0
  mfem::VectorFunctionCoefficient zero_bc(dim, [](const mfem::Vector& x, mfem::Vector& X) {
    X = 0.;
    for (int i = 0; i < X.Size(); i++)
      if (std::abs(x[i]) < 1.e-13) {
        X[i] = 1.;
      }
  });

  mfem::Array<int> ess_corner_bc_list = mfem_ext::MakeTrueEssList(solid_solver.displacement().space(), zero_bc);

  solid_solver.setTrueDofs(ess_corner_bc_list, disp_coef, *disp_bc.coef_opts.component);

  // Setup glvis output
  solid_solver.initializeOutput(serac::OutputType::GLVis, "component_bc");

  // Complete the solver setup
  solid_solver.completeSetup();

  double dt = inlet["dt"];
  solid_solver.advanceTimestep(dt);

  // Output the state
  solid_solver.outputState();

  auto state = solid_solver.getState();

  double x_norm = solid_solver.displacement().norm();

  EXPECT_NEAR(inlet["expected_u_l2norm"], x_norm, inlet["epsilon"]);
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

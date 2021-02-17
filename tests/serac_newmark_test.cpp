// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// # Author: Jonathan Wong @ LLNL.

#include <gtest/gtest.h>

#include <memory>

#include "serac/coefficients/coefficient_extensions.hpp"
#include "../src/serac/integrators/wrapper_integrator.hpp"
#include "../src/serac/numerics/expr_template_ops.hpp"
#include "mfem.hpp"

#include "serac/infrastructure/input.hpp"
#include "serac/serac_config.hpp"
#include "test_utilities.hpp"
#include "serac/physics/nonlinear_solid.hpp"
#include "serac/numerics/mesh_utils.hpp"
#include "serac/physics/operators/odes.hpp"

using namespace std;

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  int return_code = RUN_ALL_TESTS();
  MPI_Finalize();
  return return_code;
}

class NewmarkBetaTest : public ::testing::Test {
protected:
  void SetUp() {}

  void TearDown() {}

  // Helper method to run serac_newmark_tests
  std::unique_ptr<serac::NonlinearSolid> runDynamicTest(axom::inlet::Inlet& inlet, const std::string& root_name)
  {
    // Define schema
    // Simulation time parameters
    inlet.addDouble("dt", "Time step.");
    inlet.addDouble("t_final", "Stopping point");

    // Integration test parameters
    inlet.addDouble("epsilon", "Threshold to be used in the comparison");

    auto& mesh_table = inlet.addStruct("main_mesh", "The main mesh for the problem");
    serac::mesh::InputOptions::defineInputFileSchema(mesh_table);

    // Physics
    auto& solid_solver_table = inlet.addStruct("nonlinear_solid", "Finite deformation solid mechanics module");
    // FIXME: Remove once Inlet's "contains" logic improvements are merged
    serac::NonlinearSolid::InputOptions::defineInputFileSchema(solid_solver_table);
    // get gravity parameter for this problem
    inlet.addDouble("g", "the gravity acceleration");

    // Verify input file
    if (!inlet.verify()) {
      SLIC_ERROR("Input file failed to verify.");
    }

    // Build Mesh
    auto       mesh_options = inlet["main_mesh"].get<serac::mesh::InputOptions>();
    const auto rect_options = std::get_if<serac::mesh::GenerateInputOptions>(&mesh_options.extra_options);
    auto       pmesh        = serac::buildRectangleMesh(*rect_options);

    // Define the solid solver object
    auto solid_solver_options = inlet["nonlinear_solid"].get<serac::NonlinearSolid::InputOptions>();

    // We only want to add these boundary conditions if we've defined boundary_conds for the serac_newmark_beta test
    if (inlet["nonlinear_solid"].contains("boundary_conds")) {
      int                       ne = rect_options->elements[0];
      mfem::FunctionCoefficient fixed([ne](const mfem::Vector& x) { return (x[0] < 1. / ne) ? 1. : 0.; });

      mfem::Array<int> bdr_attr_list = serac::mfem_ext::MakeBdrAttributeList(*pmesh, fixed);
      for (int be = 0; be < pmesh->GetNBE(); be++) {
        pmesh->GetBdrElement(be)->SetAttribute(bdr_attr_list[be]);
      }
      pmesh->SetAttributes();
    }
    const int space_dim = pmesh->SpaceDimension();
    serac::StateManager::setMesh(std::move(pmesh));

    auto solid_solver = std::make_unique<serac::NonlinearSolid>(solid_solver_options);

    const bool is_dynamic = inlet["nonlinear_solid"].contains("dynamics");

    if (is_dynamic) {
      auto visc = std::make_unique<mfem::ConstantCoefficient>(0.0);
      solid_solver->setViscosity(std::move(visc));
    }

    // add gravity load
    if (inlet.contains("g")) {
      mfem::Vector gravity(space_dim);
      gravity    = 0.;
      gravity[1] = inlet["g"];
      solid_solver->addBodyForce(std::make_shared<mfem::VectorConstantCoefficient>(gravity));
    }

    // Initialize the output
    solid_solver->initializeOutput(serac::OutputType::VisIt, root_name);

    // Complete the solver setup
    solid_solver->completeSetup();
    // Output the initial state
    solid_solver->outputState();

    return solid_solver;
  }
};

TEST_F(NewmarkBetaTest, SimpleLua)
{
  double beta = 0.25, gamma = 0.5;

  // Create DataStore
  axom::sidre::DataStore datastore;
  // Intialize MFEMSidreDataCollection
  serac::StateManager::initialize(datastore);

  // Initialize Inlet and read input file
  std::string input_file =
      std::string(SERAC_REPO_DIR) + "/data/input_files/tests/nonlinear_solid/dyn_newmark_solve.lua";
  std::cout << input_file << std::endl;
  auto inlet = serac::input::initialize(datastore, input_file);

  auto solid_solver = runDynamicTest(inlet, "nonlin_solid_simple");

  // Save initial state
  mfem::Vector u_prev(solid_solver->displacement().gridFunc());
  mfem::Vector v_prev(solid_solver->velocity().gridFunc());

  double dt = inlet["dt"];

  // Check if dynamic
  if (inlet["nonlinear_solid"].contains("dynamics")) {
    double t       = 0.0;
    double t_final = inlet["t_final"];

    // Perform time-integration
    // (looping over the time iterations, ti, with a time-step dt).
    bool last_step = false;
    for (int ti = 1; !last_step; ti++) {
      double dt_real = std::min(dt, t_final - t);
      t += dt_real;
      last_step = (t >= t_final - 1e-8 * dt);

      solid_solver->advanceTimestep(dt_real);

      solid_solver->outputState();
    }
  } else {
    solid_solver->advanceTimestep(dt);
  }

  // Output the final state
  solid_solver->outputState();

  mfem::Vector u_next(solid_solver->displacement().gridFunc());
  mfem::Vector v_next(solid_solver->velocity().gridFunc());

  // back out a_next
  mfem::Vector a_prev(u_next.Size());
  a_prev              = 0.;
  mfem::Vector a_next = (v_next - v_prev) / (dt * gamma);

  u_next.Print();
  v_next.Print();

  double epsilon = inlet["epsilon"];

  // Check udot
  for (int d = 0; d < u_next.Size(); d++)
    EXPECT_NEAR(u_next[d],
                u_prev[d] + dt * v_prev[d] + 0.5 * dt * dt * ((1. - 2. * beta) * a_prev[d] + 2. * beta * a_next[d]),
                std::max(1.e-4 * u_next[d], epsilon));

  // Check vdot
  for (int d = 0; d < v_next.Size(); d++)
    EXPECT_NEAR(v_next[d], v_prev[d] + dt * (1 - gamma) * a_prev[d] + gamma * dt * a_next[d],
                std::max(1.e-4 * v_next[d], epsilon));
}

TEST_F(NewmarkBetaTest, EquilbriumLua)
{
  // Create DataStore
  axom::sidre::DataStore datastore;
  // Intialize MFEMSidreDataCollection
  serac::StateManager::initialize(datastore);

  // Initialize Inlet and read input file
  std::string input_file =
      std::string(SERAC_REPO_DIR) + "/data/input_files/tests/nonlinear_solid/dyn_newmark_solve_bending.lua";
  std::cout << input_file << std::endl;
  auto inlet = serac::input::initialize(datastore, input_file);

  // User helper to run test
  auto solid_solver = runDynamicTest(inlet, "nonlin_solid");

  double dt = inlet["dt"];

  // Check if dynamic
  if (inlet["nonlinear_solid"].contains("dynamics")) {
    double t       = 0.0;
    double t_final = inlet["t_final"];

    // Perform time-integration
    // (looping over the time iterations, ti, with a time-step dt).
    bool last_step = false;
    for (int ti = 1; !last_step; ti++) {
      double dt_real = std::min(dt, t_final - t);
      t += dt_real;
      last_step = (t >= t_final - 1e-8 * dt);

      solid_solver->advanceTimestep(dt_real);

      solid_solver->outputState();
    }
  } else {
    solid_solver->advanceTimestep(dt);
  }

  // Output the final state
  solid_solver->outputState();
}

TEST_F(NewmarkBetaTest, FirstOrderEquilbriumLua)
{
  // Create DataStore
  axom::sidre::DataStore datastore;
  // Intialize MFEMSidreDataCollection
  serac::StateManager::initialize(datastore);

  // Initialize Inlet and read input file
  std::string input_file =
      std::string(SERAC_REPO_DIR) + "/data/input_files/tests/nonlinear_solid/dyn_newmark_solve_bending_first.lua";
  std::cout << input_file << std::endl;
  auto inlet = serac::input::initialize(datastore, input_file);

  // User helper to run test
  auto solid_solver = runDynamicTest(inlet, "nonlin_solid_first_orderlua");

  double dt = inlet["dt"];

  // Check if dynamic
  if (inlet["nonlinear_solid"].contains("dynamics")) {
    double t       = 0.0;
    double t_final = inlet["t_final"];

    // Perform time-integration
    // (looping over the time iterations, ti, with a time-step dt).
    bool last_step = false;
    for (int ti = 1; !last_step; ti++) {
      double dt_real = std::min(dt, t_final - t);
      t += dt_real;
      last_step = (t >= t_final - 1e-8 * dt);

      solid_solver->advanceTimestep(dt_real);

      solid_solver->outputState();
    }
  } else {
    solid_solver->advanceTimestep(dt);
  }

  // Output the final state
  solid_solver->outputState();
}

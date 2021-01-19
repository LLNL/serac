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
  void SetUp()
  {
    // Set up mesh
    dim = 2;
    nex = 3;
    ney = 1;

    len   = 8.;
    width = 1.;

    mfem::Mesh mesh(nex, ney, mfem::Element::QUADRILATERAL, 1, len, width);
    pmesh = std::make_shared<mfem::ParMesh>(MPI_COMM_WORLD, mesh);
    pfes  = std::make_shared<mfem::ParFiniteElementSpace>(
        pmesh.get(), new mfem::H1_FECollection(1, dim, mfem::BasisType::GaussLobatto), 1, mfem::Ordering::byNODES);
    pfes_v = std::make_shared<mfem::ParFiniteElementSpace>(
        pmesh.get(), new mfem::H1_FECollection(1, dim, mfem::BasisType::GaussLobatto), dim, mfem::Ordering::byNODES);

    pfes_l2 = std::make_shared<mfem::ParFiniteElementSpace>(pmesh.get(), new mfem::L2_FECollection(0, dim), 1,
                                                            mfem::Ordering::byNODES);
  }

  void TearDown() {}

  double                                       width, len;
  int                                          nex, ney, nez;
  int                                          dim;
  std::shared_ptr<mfem::ParMesh>               pmesh;
  std::shared_ptr<mfem::ParFiniteElementSpace> pfes;
  std::shared_ptr<mfem::ParFiniteElementSpace> pfes_v;
  std::shared_ptr<mfem::ParFiniteElementSpace> pfes_l2;
};

TEST_F(NewmarkBetaTest, SimpleLua)
{
  double beta = 0.25, gamma = 0.5;

  // Create DataStore
  axom::sidre::DataStore datastore;

  // Initialize Inlet and read input file
  std::string input_file =
      std::string(SERAC_REPO_DIR) + "/data/input_files/tests/nonlinear_solid/dyn_newmark_solve.lua";
  std::cout << input_file << std::endl;
  auto inlet = serac::input::initialize(datastore, input_file);

  // define schema
  // Simulation time parameters
  inlet.addDouble("dt", "Time step.");
  inlet.addDouble("t_final", "Stopping point");

  // Integration test parameters
  inlet.addDouble("epsilon", "Threshold to be used in the comparison");

  auto& mesh_table = inlet.addTable("main_mesh", "The main mesh for the problem");
  serac::mesh::InputOptions::defineInputFileSchema(mesh_table);

  // Physics
  auto& solid_solver_table = inlet.addTable("nonlinear_solid", "Finite deformation solid mechanics module");
  // FIXME: Remove once Inlet's "contains" logic improvements are merged

  // we copy and paste this for now.. we want to disable bc_conditions because there aren't any in this case
  serac::NonlinearSolid::InputOptions::defineInputFileSchema(solid_solver_table);

  // Verify input file
  if (!inlet.verify()) {
    SLIC_ERROR("Input file failed to verify.");
  }

  // Define the solid solver object
  auto                  solid_solver_options = inlet["nonlinear_solid"].get<serac::NonlinearSolid::InputOptions>();
  serac::NonlinearSolid solid_solver(pmesh, solid_solver_options);

  const bool is_dynamic = inlet["nonlinear_solid"].contains("dynamics");

  if (is_dynamic) {
    auto visc = std::make_unique<mfem::ConstantCoefficient>(0.0);
    solid_solver.setViscosity(std::move(visc));
  }

  // initialize in 2D
  mfem::Vector up(2);
  up[0] = 0.;
  up[1] = 1.;
  mfem::VectorConstantCoefficient up_coef(up);
  solid_solver.setVelocity(up_coef);

  // Initialize the output
  solid_solver.initializeOutput(serac::OutputType::VisIt, "nonlin_solid");

  // Complete the solver setup
  solid_solver.completeSetup();
  // Output the initial state
  solid_solver.outputState();

  mfem::Vector u_prev(solid_solver.displacement().gridFunc());
  mfem::Vector v_prev(solid_solver.velocity().gridFunc());

  double dt = inlet["dt"];

  // Check if dynamic
  if (is_dynamic) {
    double t       = 0.0;
    double t_final = inlet["t_final"];

    // Perform time-integration
    // (looping over the time iterations, ti, with a time-step dt).
    bool last_step = false;
    for (int ti = 1; !last_step; ti++) {
      double dt_real = std::min(dt, t_final - t);
      t += dt_real;
      last_step = (t >= t_final - 1e-8 * dt);

      solid_solver.advanceTimestep(dt_real);
    }
  } else {
    solid_solver.advanceTimestep(dt);
  }

  // Output the final state
  solid_solver.outputState();
  mfem::Vector u_next(solid_solver.displacement().gridFunc());
  mfem::Vector v_next(solid_solver.velocity().gridFunc());

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

  // Initialize Inlet and read input file
  std::string input_file =
      std::string(SERAC_REPO_DIR) + "/data/input_files/tests/nonlinear_solid/dyn_newmark_solve_bending.lua";
  std::cout << input_file << std::endl;
  auto inlet = serac::input::initialize(datastore, input_file);

  // define schema
  // Simulation time parameters
  inlet.addDouble("dt", "Time step.");
  inlet.addDouble("t_final", "Stopping point");

  // Integration test parameters
  inlet.addDouble("epsilon", "Threshold to be used in the comparison");

  auto& mesh_table = inlet.addTable("main_mesh", "The main mesh for the problem");
  serac::mesh::InputOptions::defineInputFileSchema(mesh_table);

  // Physics
  auto& solid_solver_table = inlet.addTable("nonlinear_solid", "Finite deformation solid mechanics module");
  // FIXME: Remove once Inlet's "contains" logic improvements are merged
  serac::NonlinearSolid::InputOptions::defineInputFileSchema(solid_solver_table);
  
  // get gravity parameter for this problem
  inlet.addDouble("g", "the gravity acceleration");

  // Verify input file
  if (!inlet.verify()) {
    SLIC_ERROR("Input file failed to verify.");
  }

  // Define the solid solver object
  auto solid_solver_options = inlet["nonlinear_solid"].get<serac::NonlinearSolid::InputOptions>();

  int                       ne = nex;
  mfem::FunctionCoefficient fixed([ne](const mfem::Vector& x) { return (x[0] < 1. / ne) ? 1. : 0.; });

  mfem::Array<int> bdr_attr_list = serac::makeBdrAttributeList(*pmesh, fixed);
  for (int be = 0; be < pmesh->GetNBE(); be++) {
    pmesh->GetBdrElement(be)->SetAttribute(bdr_attr_list[be]);
  }
  pmesh->SetAttributes();

  serac::NonlinearSolid solid_solver(pmesh, solid_solver_options);

  const bool is_dynamic = inlet["nonlinear_solid"].contains("dynamics");

  if (is_dynamic) {
    auto visc = std::make_unique<mfem::ConstantCoefficient>(0.0);
    solid_solver.setViscosity(std::move(visc));
  }

  // add gravity load
  mfem::Vector gravity(dim);
  gravity    = 0.;
  gravity[1] = inlet["g"];
  solid_solver.addBodyForce(std::make_shared<mfem::VectorConstantCoefficient>(gravity));

  // Assume everything is initially at rest

  // Initialize the output
  solid_solver.initializeOutput(serac::OutputType::VisIt, "nonlin_solid");

  // Complete the solver setup
  solid_solver.completeSetup();
  // Output the initial state
  solid_solver.outputState();

  double dt = inlet["dt"];

  mfem::VisItDataCollection visit("NewmarkBetaLua", pmesh.get());
  visit.RegisterField("u_next", &solid_solver.displacement().gridFunc());
  visit.RegisterField("v_next", &solid_solver.velocity().gridFunc());
  visit.SetCycle(0);
  visit.SetTime(0.);
  visit.Save();

  // Check if dynamic
  if (is_dynamic) {
    double t       = 0.0;
    double t_final = inlet["t_final"];

    // Perform time-integration
    // (looping over the time iterations, ti, with a time-step dt).
    bool last_step = false;
    int  nsteps    = 0;
    for (int ti = 1; !last_step; ti++) {
      double dt_real = std::min(dt, t_final - t);
      t += dt_real;
      last_step = (t >= t_final - 1e-8 * dt);

      solid_solver.advanceTimestep(dt_real);

      solid_solver.outputState();
      visit.SetTime(t);
      visit.SetCycle(++nsteps);
      visit.Save();
    }
  } else {
    solid_solver.advanceTimestep(dt);
  }

  // Output the final state
  solid_solver.outputState();
}

TEST_F(NewmarkBetaTest, FirstOrderEquilbriumLua)
{
  
  // Create DataStore
  axom::sidre::DataStore datastore;

  // Initialize Inlet and read input file
  std::string input_file = std::string(SERAC_REPO_DIR) + "/data/input_files/tests/nonlinear_solid/dyn_newmark_solve_bending_first.lua";
  std::cout << input_file << std::endl;
  auto inlet = serac::input::initialize(datastore, input_file);

  // define schema
  // Simulation time parameters
  inlet.addDouble("dt", "Time step.");
  inlet.addDouble("t_final", "Stopping point");

  // Integration test parameters
  inlet.addDouble("epsilon", "Threshold to be used in the comparison");

  auto& mesh_table = inlet.addTable("main_mesh", "The main mesh for the problem");
  serac::mesh::InputOptions::defineInputFileSchema(mesh_table);

  // Physics
  auto& solid_solver_table = inlet.addTable("nonlinear_solid", "Finite deformation solid mechanics module");
  // FIXME: Remove once Inlet's "contains" logic improvements are merged

  
  // serac::NonlinearSolid::InputOptions::defineInputFileSchema(solid_solver_table);
  // Polynomial interpolation order - currently up to 8th order is allowed
  solid_solver_table.addInt("order", "Order degree of the finite elements.").defaultValue(1).range(1, 8);

  // neo-Hookean material parameters
  solid_solver_table.addDouble("mu", "Shear modulus in the Neo-Hookean hyperelastic model.").defaultValue(0.25);
  solid_solver_table.addDouble("K", "Bulk modulus in the Neo-Hookean hyperelastic model.").defaultValue(5.0);

  auto& stiffness_solver_solid_solver_table =
      solid_solver_table.addTable("stiffness_solver", "Linear and Nonlinear stiffness Solver Parameters.");
  serac::EquationSolver::defineInputFileSchema(stiffness_solver_solid_solver_table);

  auto& dynamics_solid_solver_table = solid_solver_table.addTable("dynamics", "Parameters for mass matrix inversion");
  dynamics_solid_solver_table.addString("timestepper", "Timestepper (ODE) method to use");
  dynamics_solid_solver_table.addString("enforcement_method", "Time-varying constraint enforcement method to use");

  auto& bc_solid_solver_table = solid_solver_table.addGenericDictionary("boundary_conds", "Solid_Solver_Table of boundary conditions");
  serac::input::BoundaryConditionInputOptions::defineInputFileSchema(bc_solid_solver_table);

  auto& init_displ = solid_solver_table.addTable("initial_displacement", "Coefficient for initial condition");
  serac::input::CoefficientInputOptions::defineInputFileSchema(init_displ);
  auto& init_velo = solid_solver_table.addTable("initial_velocity", "Coefficient for initial condition");
  serac::input::CoefficientInputOptions::defineInputFileSchema(init_velo);
  
  // get gravity parameter for this problem
  inlet.addDouble("g", "the gravity acceleration");
  
  // Verify input file
  if (!inlet.verify()) {
    SLIC_ERROR("Input file failed to verify.");
  } 
  
  // Define the solid solver object
  auto solid_solver_options = inlet["nonlinear_solid"].get<serac::NonlinearSolid::InputOptions>();

  int                           ne = nex;
  mfem::FunctionCoefficient fixed([ne](const mfem::Vector& x) { return (x[0] < 1. / ne) ? 1. : 0.; });

  mfem::Array<int> bdr_attr_list = serac::makeBdrAttributeList(*pmesh, fixed);
  for (int be = 0; be < pmesh->GetNBE(); be++) {
    pmesh->GetBdrElement(be)->SetAttribute(bdr_attr_list[be]);
  }
  pmesh->SetAttributes();
  
  serac::NonlinearSolid solid_solver(pmesh, solid_solver_options);
  
  const bool is_dynamic = inlet["nonlinear_solid"].contains("dynamics");

  if (is_dynamic) {
    auto visc = std::make_unique<mfem::ConstantCoefficient>(0.0);
    solid_solver.setViscosity(std::move(visc));
  }

  // add gravity load
  mfem::Vector gravity(dim);
  gravity = 0.;
  gravity[1] = inlet["g"];
  solid_solver.addBodyForce(std::make_shared<mfem::VectorConstantCoefficient>(gravity));
  
  // Assume everything is initially at rest
  
  // Initialize the output
  solid_solver.initializeOutput(serac::OutputType::VisIt, "nonlin_solid_first_orderlua");

  // Complete the solver setup
  solid_solver.completeSetup();
  // Output the initial state
  solid_solver.outputState();
  
  double dt = inlet["dt"]; 
  
  mfem::VisItDataCollection visit("FirsOrderLua", pmesh.get());
  visit.RegisterField("u_next", &solid_solver.displacement().gridFunc());
  visit.RegisterField("v_next", &solid_solver.velocity().gridFunc());
  visit.SetCycle(0);
  visit.SetTime(0.);
  visit.Save();
  
  // Check if dynamic
  if (is_dynamic) {
    double t       = 0.0;
    double t_final = inlet["t_final"];

    // Perform time-integration
    // (looping over the time iterations, ti, with a time-step dt).
    bool last_step = false;
    int nsteps = 0;
    for (int ti = 1; !last_step; ti++) {
      double dt_real = std::min(dt, t_final - t);
      t += dt_real;
      last_step = (t >= t_final - 1e-8 * dt);

      solid_solver.advanceTimestep(dt_real);

      solid_solver.outputState();
      visit.SetTime(t);
      visit.SetCycle( ++nsteps );
      visit.Save();
    }
  } else {
    solid_solver.advanceTimestep(dt);
  }
  

  // Output the final state
  solid_solver.outputState();
  
}

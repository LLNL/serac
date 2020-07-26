// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>
#include <sys/stat.h>

#include <fstream>

#include "mfem.hpp"
#include "serac_config.hpp"
#include "solvers/thermal_solver.hpp"

double BoundaryTemperature(const mfem::Vector& x) { return x.Norml2(); }
double OtherBoundaryTemperature(const mfem::Vector& x) { return 2 * x.Norml2(); }

double InitialTemperature(const mfem::Vector& x)
{
  if (x.Norml2() < 0.5) {
    return 2.0;
  } else {
    return 1.0;
  }
}

TEST(thermal_solver, static_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Open the mesh
  std::string  mesh_file = std::string(SERAC_REPO_DIR) + "/data/star.mesh";
  std::fstream imesh(mesh_file);
  auto         mesh = std::make_unique<mfem::Mesh>(imesh, 1, 1, true);
  imesh.close();

  // Refine in serial
  mesh->UniformRefinement();

  // Initialize the parallel mesh and delete the serial mesh
  auto pmesh = std::make_shared<mfem::ParMesh>(MPI_COMM_WORLD, *mesh);

  // Refine the parallel mesh
  pmesh->UniformRefinement();

  // Initialize the second order thermal solver on the parallel mesh
  ThermalSolver therm_solver(2, pmesh);

  // Set the time integration method
  therm_solver.setTimestepper(serac::TimestepMethod::QuasiStatic);

  // Initialize the temperature boundary condition
  auto u_0 = std::make_shared<mfem::FunctionCoefficient>(BoundaryTemperature);

  std::set<int> temp_bdr = {1};

  // Set the temperature BC in the thermal solver
  therm_solver.setTemperatureBCs(temp_bdr, u_0);

  // Set the conductivity of the thermal operator
  auto kappa = std::make_shared<mfem::ConstantCoefficient>(0.5);
  therm_solver.setConductivity(kappa);

  // Define the linear solver params
  serac::LinearSolverParameters params;
  params.rel_tol     = 1.0e-6;
  params.abs_tol     = 1.0e-12;
  params.print_level = 0;
  params.max_iter    = 100;
  therm_solver.setLinearSolverParameters(params);

  // Complete the setup without allocating the mass matrices and dynamic
  // operator
  therm_solver.completeSetup();

  // Perform the static solve
  double dt = 1.0;
  therm_solver.advanceTimestep(dt);

  // Measure the L2 norm of the solution and check the value
  mfem::ConstantCoefficient zero(0.0);
  double                    u_norm = therm_solver.getTemperature()->gf->ComputeLpError(2.0, zero);
  EXPECT_NEAR(2.56980679, u_norm, 0.00001);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(thermal_solver, static_solve_multiple_bcs)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Open the mesh
  std::string  mesh_file = std::string(SERAC_REPO_DIR) + "/data/star_with_2_bdr_attributes.mesh";
  std::fstream imesh(mesh_file);

  auto mesh = std::make_unique<mfem::Mesh>(imesh, 1, 1, true);
  imesh.close();

  // Refine in serial
  mesh->UniformRefinement();

  // Initialize the parallel mesh and delete the serial mesh
  auto pmesh = std::make_shared<mfem::ParMesh>(MPI_COMM_WORLD, *mesh);

  // Refine the parallel mesh
  pmesh->UniformRefinement();

  // Initialize the second order thermal solver on the parallel mesh
  ThermalSolver therm_solver(2, pmesh);

  // Set the time integration method
  therm_solver.setTimestepper(serac::TimestepMethod::QuasiStatic);

  // Initialize the temperature boundary condition
  auto u_0 = std::make_shared<mfem::FunctionCoefficient>(BoundaryTemperature);
  auto u_1 = std::make_shared<mfem::ConstantCoefficient>(0.0);

  std::set<int> marked_1 = {1};
  std::set<int> marked_2 = {2};

  // Set the temperature BC in the thermal solver
  therm_solver.setTemperatureBCs(marked_1, u_0);
  therm_solver.setTemperatureBCs(marked_2, u_1);

  // Set the conductivity of the thermal operator
  auto kappa = std::make_shared<mfem::ConstantCoefficient>(0.5);
  therm_solver.setConductivity(kappa);

  // Define the linear solver params
  serac::LinearSolverParameters params;
  params.rel_tol     = 1.0e-6;
  params.abs_tol     = 1.0e-12;
  params.print_level = 0;
  params.max_iter    = 100;
  therm_solver.setLinearSolverParameters(params);

  // Complete the setup without allocating the mass matrices and dynamic
  // operator
  therm_solver.completeSetup();

  // Initialize the output
  therm_solver.initializeOutput(serac::OutputType::GLVis, "thermal_two_boundary");

  // Perform the static solve
  double dt = 1.0;
  therm_solver.advanceTimestep(dt);

  // Output the state
  therm_solver.outputState();

  // Measure the L2 norm of the solution and check the value
  mfem::ConstantCoefficient zero(0.0);
  double                    u_norm = therm_solver.getTemperature()->gf->ComputeLpError(2.0, zero);
  EXPECT_NEAR(0.9168086318, u_norm, 0.00001);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(thermal_solver, static_solve_repeated_bcs)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Open the mesh
  std::string  mesh_file = std::string(SERAC_REPO_DIR) + "/data/star.mesh";
  std::fstream imesh(mesh_file);

  auto mesh = std::make_unique<mfem::Mesh>(imesh, 1, 1, true);
  imesh.close();

  // Refine in serial
  mesh->UniformRefinement();

  // Initialize the parallel mesh and delete the serial mesh
  auto pmesh = std::make_shared<mfem::ParMesh>(MPI_COMM_WORLD, *mesh);

  // Refine the parallel mesh
  pmesh->UniformRefinement();

  // Initialize the second order thermal solver on the parallel mesh
  ThermalSolver therm_solver(2, pmesh);

  // Set the time integration method
  therm_solver.setTimestepper(serac::TimestepMethod::QuasiStatic);

  // Initialize the temperature boundary condition
  auto u_0 = std::make_shared<mfem::FunctionCoefficient>(BoundaryTemperature);
  auto u_1 = std::make_shared<mfem::FunctionCoefficient>(OtherBoundaryTemperature);

  std::set<int> temp_bdr = {1};

  // Set the temperature BC in the thermal solver
  therm_solver.setTemperatureBCs(temp_bdr, u_0);
  therm_solver.setTemperatureBCs(temp_bdr, u_1);

  // Set the conductivity of the thermal operator
  auto kappa = std::make_shared<mfem::ConstantCoefficient>(0.5);
  therm_solver.setConductivity(kappa);

  // Define the linear solver params
  serac::LinearSolverParameters params;
  params.rel_tol     = 1.0e-6;
  params.abs_tol     = 1.0e-12;
  params.print_level = 0;
  params.max_iter    = 100;
  therm_solver.setLinearSolverParameters(params);

  // Complete the setup without allocating the mass matrices and dynamic
  // operator
  therm_solver.completeSetup();

  // Perform the static solve
  double dt = 1.0;
  therm_solver.advanceTimestep(dt);

  // Measure the L2 norm of the solution and check the value
  mfem::ConstantCoefficient zero(0.0);
  double                    u_norm = therm_solver.getTemperature()->gf->ComputeLpError(2.0, zero);
  EXPECT_NEAR(2.56980679, u_norm, 0.00001);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(thermal_solver, dyn_exp_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Open the mesh
  std::string  mesh_file = std::string(SERAC_REPO_DIR) + "/data/star.mesh";
  std::fstream imesh(mesh_file);

  auto mesh = std::make_unique<mfem::Mesh>(imesh, 1, 1, true);
  imesh.close();

  // Refine in serial
  mesh->UniformRefinement();

  // Initialize the parallel mesh and delete the serial mesh
  auto pmesh = std::make_shared<mfem::ParMesh>(MPI_COMM_WORLD, *mesh);

  // Refine the parallel mesh
  pmesh->UniformRefinement();

  // Initialize the second order thermal solver on the parallel mesh
  ThermalSolver therm_solver(2, pmesh);

  // Set the time integration method
  therm_solver.setTimestepper(serac::TimestepMethod::ForwardEuler);

  // Initialize the state grid function
  auto u_0 = std::make_shared<mfem::FunctionCoefficient>(InitialTemperature);
  therm_solver.setTemperature(*u_0);

  std::set<int> temp_bdr = {1};
  therm_solver.setTemperatureBCs(temp_bdr, u_0);

  // Set the conductivity of the thermal operator
  auto kappa = std::make_shared<mfem::ConstantCoefficient>(0.5);
  therm_solver.setConductivity(kappa);

  // Define the linear solver params
  serac::LinearSolverParameters params;
  params.rel_tol     = 1.0e-6;
  params.abs_tol     = 1.0e-12;
  params.print_level = 0;
  params.max_iter    = 100;
  therm_solver.setLinearSolverParameters(params);

  // Setup glvis output
  therm_solver.initializeOutput(serac::OutputType::GLVis, "thermal_explicit");

  // Complete the setup including the dynamic operators
  therm_solver.completeSetup();

  // Set timestep options
  double t         = 0.0;
  double t_final   = 0.001;
  double dt        = 0.0001;
  bool   last_step = false;

  // Output the initial state
  therm_solver.outputState();

  for (int ti = 1; !last_step; ti++) {
    double dt_real = std::min(dt, t_final - t);
    t += dt_real;
    last_step = (t >= t_final - 1e-8 * dt);

    // Advance the timestep
    therm_solver.advanceTimestep(dt_real);
  }

  // Output the final state
  therm_solver.outputState();

  // Measure the L2 norm of the solution and check the value
  mfem::ConstantCoefficient zero(0.0);
  double                    u_norm = therm_solver.getTemperature()->gf->ComputeLpError(2.0, zero);
  EXPECT_NEAR(2.6493029, u_norm, 0.00001);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(thermal_solver, dyn_imp_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Open the mesh
  std::string  mesh_file = std::string(SERAC_REPO_DIR) + "/data/star.mesh";
  std::fstream imesh(mesh_file);
  auto         mesh = std::make_unique<mfem::Mesh>(imesh, 1, 1, true);
  imesh.close();

  // Refine in serial
  mesh->UniformRefinement();

  // Initialize the parallel mesh and delete the serial mesh
  auto pmesh = std::make_shared<mfem::ParMesh>(MPI_COMM_WORLD, *mesh);

  // Refine the parallel mesh
  pmesh->UniformRefinement();

  // Initialize the second order thermal solver on the parallel mesh
  ThermalSolver therm_solver(2, pmesh);

  // Set the time integration method
  therm_solver.setTimestepper(serac::TimestepMethod::BackwardEuler);

  // Initialize the state grid function
  auto u_0 = std::make_shared<mfem::FunctionCoefficient>(InitialTemperature);
  therm_solver.setTemperature(*u_0);

  std::set<int> temp_bdr = {1};
  therm_solver.setTemperatureBCs(temp_bdr, u_0);

  // Set the conductivity of the thermal operator
  auto kappa = std::make_shared<mfem::ConstantCoefficient>(0.5);
  therm_solver.setConductivity(kappa);

  // Define the linear solver params
  serac::LinearSolverParameters params;
  params.rel_tol     = 1.0e-6;
  params.abs_tol     = 1.0e-12;
  params.print_level = 0;
  params.max_iter    = 100;
  therm_solver.setLinearSolverParameters(params);

  // Setup glvis output
  therm_solver.initializeOutput(serac::OutputType::VisIt, "thermal_implicit");

  // Complete the setup including the dynamic operators
  therm_solver.completeSetup();

  // Set timestep options
  double t         = 0.0;
  double t_final   = 5.0;
  double dt        = 1.0;
  bool   last_step = false;

  // Output the initial state
  therm_solver.outputState();

  for (int ti = 1; !last_step; ti++) {
    double dt_real = std::min(dt, t_final - t);
    t += dt_real;
    last_step = (t >= t_final - 1e-8 * dt);

    // Advance the timestep
    therm_solver.advanceTimestep(dt_real);
  }

  // Output the final state
  therm_solver.outputState();

  // Measure the L2 norm of the solution and check the value
  mfem::ConstantCoefficient zero(0.0);
  double                    u_norm = therm_solver.getTemperature()->gf->ComputeLpError(2.0, zero);
  EXPECT_NEAR(2.18201099, u_norm, 0.00001);

  MPI_Barrier(MPI_COMM_WORLD);
}

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

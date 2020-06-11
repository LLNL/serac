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

  // mesh
  std::string base_mesh_file = std::string(SERAC_SRC_DIR) + "/data/star.mesh";
  const char* mesh_file      = base_mesh_file.c_str();

  // Open the mesh
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
  therm_solver.SetTimestepper(TimestepMethod::QuasiStatic);

  // Initialize the temperature boundary condition
  auto u_0 = std::make_shared<mfem::FunctionCoefficient>(BoundaryTemperature);

  std::vector<int> temp_bdr(pmesh->bdr_attributes.Max(), 1);

  // Set the temperature BC in the thermal solver
  therm_solver.SetTemperatureBCs(temp_bdr, u_0);

  // Set the conductivity of the thermal operator
  auto kappa = std::make_shared<mfem::ConstantCoefficient>(0.5);
  therm_solver.SetConductivity(kappa);

  // Define the linear solver params
  LinearSolverParameters params;
  params.rel_tol     = 1.0e-6;
  params.abs_tol     = 1.0e-12;
  params.print_level = 0;
  params.max_iter    = 100;
  therm_solver.SetLinearSolverParameters(params);

  // Complete the setup without allocating the mass matrices and dynamic
  // operator
  therm_solver.CompleteSetup();

  // Perform the static solve
  double dt = 1.0;
  therm_solver.AdvanceTimestep(dt);

  // Get the state grid function
  auto state = therm_solver.GetState();

  // Measure the L2 norm of the solution and check the value
  mfem::ConstantCoefficient zero(0.0);
  double                    u_norm = state[0].gf->ComputeLpError(2.0, zero);
  EXPECT_NEAR(2.56980679, u_norm, 0.00001);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(thermal_solver, static_solve_multiple_bcs)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // mesh
  std::string base_mesh_file = std::string(SERAC_SRC_DIR) + "/data/star.mesh";
  const char* mesh_file      = base_mesh_file.c_str();

  // Open the mesh
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
  therm_solver.SetTimestepper(TimestepMethod::QuasiStatic);

  // Initialize the temperature boundary condition
  auto u_0 = std::make_shared<mfem::FunctionCoefficient>(BoundaryTemperature);

  std::vector<int> marked_1(pmesh->bdr_attributes.Max(), 1);
  std::vector<int> marked_2(pmesh->bdr_attributes.Max(), 2);

  // Set the temperature BC in the thermal solver
  therm_solver.SetTemperatureBCs(marked_1, u_0);
  therm_solver.SetTemperatureBCs(marked_2, u_0);

  // Set the conductivity of the thermal operator
  auto kappa = std::make_shared<mfem::ConstantCoefficient>(0.5);
  therm_solver.SetConductivity(kappa);

  // Define the linear solver params
  LinearSolverParameters params;
  params.rel_tol     = 1.0e-6;
  params.abs_tol     = 1.0e-12;
  params.print_level = 0;
  params.max_iter    = 100;
  therm_solver.SetLinearSolverParameters(params);

  // Complete the setup without allocating the mass matrices and dynamic
  // operator
  therm_solver.CompleteSetup();

  // Perform the static solve
  double dt = 1.0;
  therm_solver.AdvanceTimestep(dt);

  // Get the state grid function
  auto state = therm_solver.GetState();

  // Measure the L2 norm of the solution and check the value
  mfem::ConstantCoefficient zero(0.0);
  double                    u_norm = state[0].gf->ComputeLpError(2.0, zero);
  EXPECT_NEAR(2.56980679, u_norm, 0.00001);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(thermal_solver, static_solve_repeated_bcs)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // mesh
  std::string base_mesh_file = std::string(SERAC_SRC_DIR) + "/data/star.mesh";
  const char* mesh_file      = base_mesh_file.c_str();

  // Open the mesh
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
  therm_solver.SetTimestepper(TimestepMethod::QuasiStatic);

  // Initialize the temperature boundary condition
  auto u_0 = std::make_shared<mfem::FunctionCoefficient>(BoundaryTemperature);
  auto u_1 = std::make_shared<mfem::FunctionCoefficient>(OtherBoundaryTemperature);

  std::vector<int> temp_bdr(pmesh->bdr_attributes.Max(), 1);

  // Set the temperature BC in the thermal solver
  therm_solver.SetTemperatureBCs(temp_bdr, u_1);
  therm_solver.SetTemperatureBCs(temp_bdr, u_0);

  // Set the conductivity of the thermal operator
  auto kappa = std::make_shared<mfem::ConstantCoefficient>(0.5);
  therm_solver.SetConductivity(kappa);

  // Define the linear solver params
  LinearSolverParameters params;
  params.rel_tol     = 1.0e-6;
  params.abs_tol     = 1.0e-12;
  params.print_level = 0;
  params.max_iter    = 100;
  therm_solver.SetLinearSolverParameters(params);

  // Complete the setup without allocating the mass matrices and dynamic
  // operator
  therm_solver.CompleteSetup();

  // Perform the static solve
  double dt = 1.0;
  therm_solver.AdvanceTimestep(dt);

  // Get the state grid function
  auto state = therm_solver.GetState();

  // Measure the L2 norm of the solution and check the value
  mfem::ConstantCoefficient zero(0.0);
  double                    u_norm = state[0].gf->ComputeLpError(2.0, zero);
  EXPECT_NEAR(2.56980679, u_norm, 0.00001);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(thermal_solver, dyn_exp_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // mesh
  std::string base_mesh_file = std::string(SERAC_SRC_DIR) + "/data/star.mesh";
  const char* mesh_file      = base_mesh_file.c_str();

  // Open the mesh
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
  therm_solver.SetTimestepper(TimestepMethod::ForwardEuler);

  // Initialize the state grid function
  auto u_0 = std::make_shared<mfem::FunctionCoefficient>(InitialTemperature);
  therm_solver.SetTemperature(*u_0);

  // Set the temperature BC in the thermal solver
  std::vector<int> temp_bdr(pmesh->bdr_attributes.Max(), 1);
  therm_solver.SetTemperatureBCs(temp_bdr, u_0);

  // Set the conductivity of the thermal operator
  auto kappa = std::make_shared<mfem::ConstantCoefficient>(0.5);
  therm_solver.SetConductivity(kappa);

  // Define the linear solver params
  LinearSolverParameters params;
  params.rel_tol     = 1.0e-6;
  params.abs_tol     = 1.0e-12;
  params.print_level = 0;
  params.max_iter    = 100;
  therm_solver.SetLinearSolverParameters(params);

  // Setup glvis output
  therm_solver.InitializeOutput(OutputType::GLVis, "thermal_explicit");

  // Complete the setup including the dynamic operators
  therm_solver.CompleteSetup();

  // Set timestep options
  double t         = 0.0;
  double t_final   = 0.001;
  double dt        = 0.0001;
  bool   last_step = false;

  // Output the initial state
  therm_solver.OutputState();

  for (int ti = 1; !last_step; ti++) {
    double dt_real = std::min(dt, t_final - t);
    t += dt_real;
    last_step = (t >= t_final - 1e-8 * dt);

    // Advance the timestep
    therm_solver.AdvanceTimestep(dt_real);
  }

  // Output the final state
  therm_solver.OutputState();

  // Get the state grid function
  auto state = therm_solver.GetState();

  // Measure the L2 norm of the solution and check the value
  mfem::ConstantCoefficient zero(0.0);
  double                    u_norm = state[0].gf->ComputeLpError(2.0, zero);
  EXPECT_NEAR(2.6493029, u_norm, 0.00001);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(thermal_solver, dyn_imp_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // mesh
  std::string base_mesh_file = std::string(SERAC_SRC_DIR) + "/data/star.mesh";
  const char* mesh_file      = base_mesh_file.c_str();

  // Open the mesh
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
  therm_solver.SetTimestepper(TimestepMethod::BackwardEuler);

  // Initialize the state grid function
  auto u_0 = std::make_shared<mfem::FunctionCoefficient>(InitialTemperature);
  therm_solver.SetTemperature(*u_0);

  // Set the temperature BC in the thermal solver
  std::vector<int> temp_bdr(pmesh->bdr_attributes.Max(), 1);
  therm_solver.SetTemperatureBCs(temp_bdr, u_0);

  // Set the conductivity of the thermal operator
  auto kappa = std::make_shared<mfem::ConstantCoefficient>(0.5);
  therm_solver.SetConductivity(kappa);

  // Define the linear solver params
  LinearSolverParameters params;
  params.rel_tol     = 1.0e-6;
  params.abs_tol     = 1.0e-12;
  params.print_level = 0;
  params.max_iter    = 100;
  therm_solver.SetLinearSolverParameters(params);

  // Setup glvis output
  therm_solver.InitializeOutput(OutputType::VisIt, "thermal_implicit");

  // Complete the setup including the dynamic operators
  therm_solver.CompleteSetup();

  // Set timestep options
  double t         = 0.0;
  double t_final   = 5.0;
  double dt        = 1.0;
  bool   last_step = false;

  // Output the initial state
  therm_solver.OutputState();

  for (int ti = 1; !last_step; ti++) {
    double dt_real = std::min(dt, t_final - t);
    t += dt_real;
    last_step = (t >= t_final - 1e-8 * dt);

    // Advance the timestep
    therm_solver.AdvanceTimestep(dt_real);
  }

  // Output the final state
  therm_solver.OutputState();

  // Get the state grid function
  auto state = therm_solver.GetState();

  // Measure the L2 norm of the solution and check the value
  mfem::ConstantCoefficient zero(0.0);
  double                    u_norm = state[0].gf->ComputeLpError(2.0, zero);
  EXPECT_NEAR(2.18201099, u_norm, 0.00001);

  MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}

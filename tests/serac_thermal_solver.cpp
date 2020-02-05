// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include "mfem.hpp"
#include "solvers/thermal_solver.hpp"
#include <fstream>
#include <sys/stat.h>

double BoundaryTemperature(const mfem::Vector &x);
double InitialTemperature(const mfem::Vector &x);

const char* mesh_file = "NO_MESH_GIVEN";

inline bool file_exists(const char* path)
{
  struct stat buffer;
  return (stat(path, &buffer) == 0);
}

TEST(thermal_solver, static_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Open the mesh
  ASSERT_TRUE(file_exists(mesh_file));
  std::fstream imesh(mesh_file);
  mfem::Mesh* mesh = new mfem::Mesh(imesh, 1, 1, true);
  imesh.close();

  // Refine in serial
  mesh->UniformRefinement();

  // Declare pointer to parallel mesh object
  mfem::ParMesh *pmesh = nullptr;

  // Initialize the parallel mesh and delete the serial mesh
  pmesh = new mfem::ParMesh(MPI_COMM_WORLD, *mesh);
  delete mesh;

  // Refine the parallel mesh
  pmesh->UniformRefinement();

  // Initialize the second order thermal solver on the parallel mesh
  ThermalSolver therm_solver(2, pmesh);

  // Set the time integration method
  therm_solver.SetTimestepper(TimestepMethod::QuasiStatic);

  // Initialize the temperature boundary condition
  mfem::FunctionCoefficient u_0(BoundaryTemperature);
  mfem::Array<int> temp_bdr(pmesh->bdr_attributes.Max());
  temp_bdr = 1;

  // Set the temperature BC in the thermal solver
  therm_solver.SetTemperatureBCs(temp_bdr, &u_0);

  // Set the conductivity of the thermal operator
  mfem::ConstantCoefficient kappa(0.5);
  therm_solver.SetConductivity(kappa);

  // Define the linear solver params
  LinearSolverParameters params;
  params.rel_tol = 1.0e-6;
  params.abs_tol = 1.0e-12;
  params.print_level = 0;
  params.max_iter = 100;
  therm_solver.SetLinearSolverParameters(params);

  // Complete the setup without allocating the mass matrices and dynamic operator
  therm_solver.CompleteSetup();

  // Perform the static solve
  double dt = 1.0;
  therm_solver.AdvanceTimestep(dt);

  // Get the state grid function
  auto state = therm_solver.GetState();

  // Measure the L2 norm of the solution and check the value
  mfem::ConstantCoefficient zero(0.0);
  double u_norm = state[0].gf->ComputeLpError(2.0, zero);
  EXPECT_NEAR(2.56980679, u_norm, 0.00001);

  delete pmesh;

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(thermal_solver, dyn_exp_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Open the mesh
  ASSERT_TRUE(file_exists(mesh_file));
  std::fstream imesh(mesh_file);
  mfem::Mesh* mesh = new mfem::Mesh(imesh, 1, 1, true);
  imesh.close();

  // Refine in serial
  mesh->UniformRefinement();

  // Declare pointer to parallel mesh object
  mfem::ParMesh *pmesh = nullptr;

  // Initialize the parallel mesh and delete the serial mesh
  pmesh = new mfem::ParMesh(MPI_COMM_WORLD, *mesh);
  delete mesh;

  // Refine the parallel mesh
  pmesh->UniformRefinement();

  // Initialize the second order thermal solver on the parallel mesh
  ThermalSolver therm_solver(2, pmesh);

  // Set the time integration method
  therm_solver.SetTimestepper(TimestepMethod::ForwardEuler);

  // Initialize the state grid function
  mfem::FunctionCoefficient u_0(InitialTemperature);
  therm_solver.SetInitialState(u_0);

  // Set the temperature BC in the thermal solver
  mfem::Array<int> temp_bdr(pmesh->bdr_attributes.Max());
  temp_bdr = 1;
  therm_solver.SetTemperatureBCs(temp_bdr, &u_0);

  // Set the conductivity of the thermal operator
  mfem::ConstantCoefficient kappa(0.5);
  therm_solver.SetConductivity(kappa);

  // Define the linear solver params
  LinearSolverParameters params;
  params.rel_tol = 1.0e-6;
  params.abs_tol = 1.0e-12;
  params.print_level = 0;
  params.max_iter = 100;
  therm_solver.SetLinearSolverParameters(params);

  // Complete the setup including the dynamic operators
  therm_solver.CompleteSetup();

  // Set timestep options
  double t = 0.0;
  double t_final = 0.001;
  double dt = 0.0001;
  bool last_step = false;

  for (int ti = 1; !last_step; ti++) {
    double dt_real = std::min(dt, t_final - t);
    last_step = (t >= t_final - 1e-8*dt);

    // Advance the timestep
    therm_solver.AdvanceTimestep(dt_real);
    t += dt_real;
  }

  // Get the state grid function
  auto state = therm_solver.GetState();

  // Measure the L2 norm of the solution and check the value
  mfem::ConstantCoefficient zero(0.0);
  double u_norm = state[0].gf->ComputeLpError(2.0, zero);
  EXPECT_NEAR(2.6493029, u_norm, 0.00001);

  delete pmesh;

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(thermal_solver, dyn_imp_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Open the mesh
  ASSERT_TRUE(file_exists(mesh_file));
  std::fstream imesh(mesh_file);
  mfem::Mesh* mesh = new mfem::Mesh(imesh, 1, 1, true);
  imesh.close();

  // Refine in serial
  mesh->UniformRefinement();

  // Declare pointer to parallel mesh object
  mfem::ParMesh *pmesh = nullptr;

  // Initialize the parallel mesh and delete the serial mesh
  pmesh = new mfem::ParMesh(MPI_COMM_WORLD, *mesh);
  delete mesh;

  // Refine the parallel mesh
  pmesh->UniformRefinement();

  // Initialize the second order thermal solver on the parallel mesh
  ThermalSolver therm_solver(2, pmesh);

  // Set the time integration method
  therm_solver.SetTimestepper(TimestepMethod::BackwardEuler);

  // Initialize the state grid function
  mfem::FunctionCoefficient u_0(InitialTemperature);
  therm_solver.SetInitialState(u_0);

  // Set the temperature BC in the thermal solver
  mfem::Array<int> temp_bdr(pmesh->bdr_attributes.Max());
  temp_bdr = 1;
  therm_solver.SetTemperatureBCs(temp_bdr, &u_0);

  // Set the conductivity of the thermal operator
  mfem::ConstantCoefficient kappa(0.5);
  therm_solver.SetConductivity(kappa);

  // Define the linear solver params
  LinearSolverParameters params;
  params.rel_tol = 1.0e-6;
  params.abs_tol = 1.0e-12;
  params.print_level = 0;
  params.max_iter = 100;
  therm_solver.SetLinearSolverParameters(params);

  // Complete the setup including the dynamic operators
  therm_solver.CompleteSetup();

  // Set timestep options
  double t = 0.0;
  double t_final = 5.0;
  double dt = 1.0;
  bool last_step = false;

  for (int ti = 1; !last_step; ti++) {
    double dt_real = std::min(dt, t_final - t);
    last_step = (t >= t_final - 1e-8*dt);

    // Advance the timestep
    therm_solver.AdvanceTimestep(dt_real);
    t += dt_real;
  }

  // Get the state grid function
  auto state = therm_solver.GetState();

  // Measure the L2 norm of the solution and check the value
  mfem::ConstantCoefficient zero(0.0);
  double u_norm = state[0].gf->ComputeLpError(2.0, zero);
  EXPECT_NEAR(2.18201099, u_norm, 0.00001);

  delete pmesh;

  MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);
  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  // Parse command line options
  mfem::OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "Mesh file to use.", true);
  args.Parse();
  if (!args.Good()) {
    if (myid == 0) {
      args.PrintUsage(std::cout);
    }
    MPI_Finalize();
    return 1;
  }
  if (myid == 0) {
    args.PrintOptions(std::cout);
  }

  result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}

double BoundaryTemperature(const mfem::Vector &x)
{
  return x.Norml2();
}

double InitialTemperature(const mfem::Vector &x)
{
  if (x.Norml2() < 0.5) {
    return 2.0;
  } else {
    return 1.0;
  }
}



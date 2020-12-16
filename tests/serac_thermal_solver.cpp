// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>
#include <sys/stat.h>

#include <fstream>

#include "mfem.hpp"
#include "serac/numerics/mesh_utils.hpp"
#include "serac/physics/thermal_conduction.hpp"
#include "serac/serac_config.hpp"

namespace serac {

double One(const mfem::Vector& /*x*/) { return 1.0; }
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

  auto pmesh = buildBallMesh(10000);

  // Initialize the second order thermal solver on the parallel mesh
  ThermalConduction therm_solver(2, pmesh, ThermalConduction::defaultQuasistaticParameters());

  // Initialize the temperature boundary condition
  auto u_0 = std::make_shared<mfem::FunctionCoefficient>(One);

  std::set<int> temp_bdr = {1};

  // Set the temperature BC in the thermal solver
  therm_solver.setTemperatureBCs(temp_bdr, u_0);

  // Set the conductivity of the thermal operator
  auto kappa = std::make_unique<mfem::ConstantCoefficient>(0.5);
  therm_solver.setConductivity(std::move(kappa));

  // Complete the setup without allocating the mass matrices and dynamic
  // operator
  therm_solver.completeSetup();

  // Perform the static solve
  double dt = 1.0;
  therm_solver.advanceTimestep(dt);

  // Measure the L2 norm of the solution and check the value
  mfem::ConstantCoefficient zero(0.0);
  double                    u_norm = therm_solver.temperature().gridFunc().ComputeLpError(2.0, zero);
  EXPECT_NEAR(2.02263, u_norm, 0.00001);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(thermal_solver, static_solve_multiple_bcs)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Open the mesh
  std::string mesh_file = std::string(SERAC_REPO_DIR) + "/data/meshes/star_with_2_bdr_attributes.mesh";

  auto pmesh = buildMeshFromFile(mesh_file, 1, 1);

  // Initialize the second order thermal solver on the parallel mesh
  ThermalConduction therm_solver(2, pmesh, ThermalConduction::defaultQuasistaticParameters());

  // Initialize the temperature boundary condition
  auto u_0 = std::make_shared<mfem::FunctionCoefficient>(BoundaryTemperature);
  auto u_1 = std::make_shared<mfem::FunctionCoefficient>(BoundaryTemperature);
  // auto u_1 = std::make_shared<mfem::ConstantCoefficient>(0.0);

  std::set<int> marked_1 = {1};
  std::set<int> marked_2 = {2};

  // Set the temperature BC in the thermal solver
  therm_solver.setTemperatureBCs(marked_1, u_0);
  therm_solver.setTemperatureBCs(marked_2, u_1);

  // Set the conductivity of the thermal operator
  auto kappa = std::make_unique<mfem::ConstantCoefficient>(0.5);
  therm_solver.setConductivity(std::move(kappa));

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
  double                    u_norm = therm_solver.temperature().gridFunc().ComputeLpError(2.0, zero);
  EXPECT_NEAR(2.56980679, u_norm, 0.00001);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(thermal_solver, static_solve_repeated_bcs)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Open the mesh
  std::string mesh_file = std::string(SERAC_REPO_DIR) + "/data/meshes/star.mesh";

  auto pmesh = buildMeshFromFile(mesh_file, 1, 1);

  // Initialize the second order thermal solver on the parallel mesh
  ThermalConduction therm_solver(2, pmesh, ThermalConduction::defaultQuasistaticParameters());

  // Initialize the temperature boundary condition
  auto u_0 = std::make_shared<mfem::FunctionCoefficient>(BoundaryTemperature);
  auto u_1 = std::make_shared<mfem::FunctionCoefficient>(OtherBoundaryTemperature);

  std::set<int> temp_bdr = {1};

  // Set the temperature BC in the thermal solver
  therm_solver.setTemperatureBCs(temp_bdr, u_0);
  therm_solver.setTemperatureBCs(temp_bdr, u_1);

  // Set the conductivity of the thermal operator
  auto kappa = std::make_unique<mfem::ConstantCoefficient>(0.5);
  therm_solver.setConductivity(std::move(kappa));

  // Complete the setup without allocating the mass matrices and dynamic
  // operator
  therm_solver.completeSetup();

  // Perform the static solve
  double dt = 1.0;
  therm_solver.advanceTimestep(dt);

  // Measure the L2 norm of the solution and check the value
  mfem::ConstantCoefficient zero(0.0);
  double                    u_norm = therm_solver.temperature().gridFunc().ComputeLpError(2.0, zero);
  EXPECT_NEAR(2.56980679, u_norm, 0.00001);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(thermal_solver, dyn_exp_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Open the mesh
  std::string mesh_file = std::string(SERAC_REPO_DIR) + "/data/meshes/star.mesh";

  auto pmesh = buildMeshFromFile(mesh_file, 1, 1);

  auto params                    = ThermalConduction::defaultDynamicParameters();
  params.dyn_params->timestepper = TimestepMethod::ForwardEuler;

  // Initialize the second order thermal solver on the parallel mesh
  ThermalConduction therm_solver(2, pmesh, params);

  // Initialize the state grid function
  auto u_0 = std::make_shared<mfem::FunctionCoefficient>(InitialTemperature);
  therm_solver.setTemperature(*u_0);

  std::set<int> temp_bdr = {1};
  therm_solver.setTemperatureBCs(temp_bdr, u_0);

  // Set the conductivity of the thermal operator
  auto kappa = std::make_unique<mfem::ConstantCoefficient>(0.5);
  therm_solver.setConductivity(std::move(kappa));

  // Setup glvis output
  therm_solver.initializeOutput(serac::OutputType::ParaView, "thermal_explicit");

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
  double                    u_norm = therm_solver.temperature().gridFunc().ComputeLpError(2.0, zero);
  EXPECT_NEAR(2.6493029, u_norm, 0.00001);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(thermal_solver, dyn_imp_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Open the mesh
  std::string mesh_file = std::string(SERAC_REPO_DIR) + "/data/meshes/star.mesh";

  auto pmesh = buildMeshFromFile(mesh_file, 1, 1);

  // Initialize the second order thermal solver on the parallel mesh
  ThermalConduction therm_solver(2, pmesh, ThermalConduction::defaultDynamicParameters());

  // Initialize the state grid function
  auto u_0 = std::make_shared<mfem::FunctionCoefficient>(InitialTemperature);
  therm_solver.setTemperature(*u_0);

  std::set<int> temp_bdr = {1};
  therm_solver.setTemperatureBCs(temp_bdr, u_0);

  // Set the density function
  auto rho = std::make_unique<mfem::ConstantCoefficient>(0.5);
  therm_solver.setDensity(std::move(rho));

  // Set the specific heat capacity function
  auto cp = std::make_unique<mfem::ConstantCoefficient>(0.5);
  therm_solver.setSpecificHeatCapacity(std::move(cp));

  // Set the conductivity of the thermal operator
  auto kappa = std::make_unique<mfem::ConstantCoefficient>(0.5);
  therm_solver.setConductivity(std::move(kappa));

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
  double                    u_norm = therm_solver.temperature().gridFunc().ComputeLpError(2.0, zero);
  EXPECT_NEAR(2.1806652643, u_norm, 0.00001);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(thermal_solver_rework, dyn_imp_solve_time_varying)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Open the mesh
  std::string mesh_file = std::string(SERAC_REPO_DIR) + "/data/meshes/star.mesh";

  auto pmesh = buildMeshFromFile(mesh_file, 2, 1);

  // Initialize the second order thermal solver on the parallel mesh
  ThermalConduction therm_solver(2, pmesh, ThermalConduction::defaultDynamicParameters());

  // by construction, f(x, y, t) satisfies df_dt == d2f_dx2 + d2f_dy2
  auto f = std::make_shared<mfem::FunctionCoefficient>([](const mfem::Vector& x, double t) {
    return 1.0 + 6.0 * x[0] * t - 2.0 * x[1] * t + (x[0] - x[1]) * x[0] * x[0];
  });

  therm_solver.setTemperature(*f);

  std::set<int> temp_bdr = {1};
  therm_solver.setTemperatureBCs(temp_bdr, f);

  // Set the conductivity of the thermal operator
  auto kappa = std::make_unique<mfem::ConstantCoefficient>(1.0);
  therm_solver.setConductivity(std::move(kappa));

  // Setup glvis output
  therm_solver.initializeOutput(serac::OutputType::VisIt, "thermal_implicit");

  // Complete the setup including the dynamic operators
  therm_solver.completeSetup();

  // Set timestep options
  double t         = 0.0;
  double t_final   = 5.0;
  double dt        = 0.5;
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

  f->SetTime(t);
  double error = therm_solver.temperature().gridFunc().ComputeLpError(2.0, *f);
  EXPECT_NEAR(0.0, error, 0.00005);

  MPI_Barrier(MPI_COMM_WORLD);
}

#ifdef MFEM_USE_AMGX
TEST(thermal_solver, static_amgx_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  auto pmesh = buildBallMesh(10000);

  auto params                                                   = ThermalConduction::defaultQuasistaticParameters();
  std::get<IterativeSolverOptions>(params.T_lin_params).prec = AMGXPrec{.smoother = AMGXSolver::JACOBI_L1};
  // Initialize the second order thermal solver on the parallel mesh
  ThermalConduction therm_solver(2, pmesh, params);

  // Initialize the temperature boundary condition
  auto u_0 = std::make_shared<mfem::FunctionCoefficient>(One);

  std::set<int> temp_bdr = {1};

  // Set the temperature BC in the thermal solver
  therm_solver.setTemperatureBCs(temp_bdr, u_0);

  // Set the conductivity of the thermal operator
  auto kappa = std::make_unique<mfem::ConstantCoefficient>(0.5);
  therm_solver.setConductivity(std::move(kappa));

  // Complete the setup without allocating the mass matrices and dynamic
  // operator
  therm_solver.completeSetup();

  // Perform the static solve
  double dt = 1.0;
  therm_solver.advanceTimestep(dt);

  // Measure the L2 norm of the solution and check the value
  mfem::ConstantCoefficient zero(0.0);
  double                    u_norm = therm_solver.temperature().gridFunc().ComputeLpError(2.0, zero);
  EXPECT_NEAR(2.02263, u_norm, 0.00001);

  MPI_Barrier(MPI_COMM_WORLD);
}
#endif

}  // namespace serac

//------------------------------------------------------------------------------
#include "axom/slic/core/UnitTestLogger.hpp"
using axom::slic::UnitTestLogger;

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  UnitTestLogger logger;  // create & initialize test logger, finalized when
                          // exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}

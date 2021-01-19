// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include "mfem.hpp"
#include "serac/numerics/mesh_utils.hpp"
#include "serac/physics/thermal_conduction.hpp"
#include "serac/serac_config.hpp"

double InitialTemperature(const mfem::Vector& x)
{
  if (x.Norml2() < 0.5) {
    return 2.0;
  } else {
    return 1.0;
  }
}

using namespace serac;

TEST(thermal_solver, dyn_imp_solve)
{
  int restart_cycle = -1;

  // Start a scope block to guarantee separation between the simulated nominal/restart runs
  {
    MPI_Barrier(MPI_COMM_WORLD);
    // Create DataStore
    axom::sidre::DataStore datastore;
    serac::StateManager::initialize(datastore);

    // Open the mesh
    std::string mesh_file = std::string(SERAC_REPO_DIR) + "/data/meshes/star.mesh";

    auto pmesh = buildMeshFromFile(mesh_file, 1, 1);
    serac::StateManager::setMesh(std::move(pmesh));

    // Initialize the second order thermal solver on the parallel mesh
    ThermalConduction therm_solver(2, ThermalConduction::defaultDynamicOptions());

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

    therm_solver.initializeOutput(serac::OutputType::SidreVisIt, "thermal_implicit");

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
      // Assign to the restart cycle
      restart_cycle = ti;
    }

    therm_solver.outputState();

    // Measure the L2 norm of the solution and check the value
    mfem::ConstantCoefficient zero(0.0);
    double                    u_norm = therm_solver.temperature().gridFunc().ComputeLpError(2.0, zero);
    EXPECT_NEAR(2.1806652643, u_norm, 0.000001);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  serac::StateManager::reset();

  // Simulate a restart
  {
    MPI_Barrier(MPI_COMM_WORLD);
    // Create DataStore
    axom::sidre::DataStore datastore;
    // Load in from the saved cycle
    serac::StateManager::initialize(datastore, restart_cycle);

    // Initialize the second order thermal solver on the parallel mesh
    ThermalConduction therm_solver(2, ThermalConduction::defaultDynamicOptions());

    auto u_0 = std::make_shared<mfem::FunctionCoefficient>(InitialTemperature);
    // Don't initialize the state grid function
    // therm_solver.setTemperature(*u_0);

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

    therm_solver.initializeOutput(serac::OutputType::SidreVisIt, "thermal_implicit");

    // Complete the setup including the dynamic operators
    therm_solver.completeSetup();

    // Set timestep options
    double t         = 5.0;
    double t_final   = 10.0;
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

    therm_solver.outputState();

    // Measure the L2 norm of the solution and check the value
    mfem::ConstantCoefficient zero(0.0);
    double                    u_norm = therm_solver.temperature().gridFunc().ComputeLpError(2.0, zero);
    // Running the initial simulation for an extra 5s produced this result, so we would expect it to be the exact same
    EXPECT_NEAR(2.1806604032633987, u_norm, 0.000001);
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

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

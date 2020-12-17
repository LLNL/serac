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
  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore);

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

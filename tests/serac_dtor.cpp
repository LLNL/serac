// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>
#include <sys/stat.h>

#include <fstream>
#include <memory>

#include "mfem.hpp"
#include "numerics/mesh_utils.hpp"
#include "physics/thermal_solver.hpp"
#include "serac_config.hpp"

namespace serac {

TEST(serac_dtor, test1)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Open the mesh
  std::string mesh_file = std::string(SERAC_REPO_DIR) + "/data/beam-hex.mesh";

  auto pmesh = buildMeshFromFile(mesh_file, 1, 0);

  // Initialize the second order thermal solver on the parallel mesh
  auto therm_solver = std::make_unique<ThermalSolver>(2, pmesh);

  // Set the time integration method
  therm_solver->setTimestepper(serac::TimestepMethod::QuasiStatic);

  // Initialize the temperature boundary condition
  mfem::FunctionCoefficient u_0([](const mfem::Vector& x) { return x.Norml2(); });

  std::set<int> temp_bdr = {1};
  // Set the temperature BC in the thermal solver
  therm_solver->setTemperatureBCs(temp_bdr, u_0);

  // Set the conductivity of the thermal operator
  mfem::ConstantCoefficient kappa(0.5);
  therm_solver->setConductivity(kappa);

  // Define the linear solver params
  serac::LinearSolverParameters params;
  params.rel_tol     = 1.0e-6;
  params.abs_tol     = 1.0e-12;
  params.print_level = 0;
  params.max_iter    = 100;
  therm_solver->setLinearSolverParameters(params);

  // Complete the setup without allocating the mass matrices and dynamic
  // operator
  therm_solver->completeSetup();

  // Destruct the old thermal solver and build a new one
  therm_solver.reset(new ThermalSolver(1, pmesh));

  // Destruct the second thermal solver and leave the pointer empty
  therm_solver.reset(nullptr);
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

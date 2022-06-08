// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <memory>
#include <sys/stat.h>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/thermal_conduction.hpp"
#include "serac/serac_config.hpp"

namespace serac {

TEST(SeracDtor, Test1)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Open the mesh
  std::string mesh_file = std::string(SERAC_REPO_DIR) + "/data/meshes/beam-hex.mesh";

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "serac_dtor");

  auto pmesh = mesh::refineAndDistribute(buildMeshFromFile(mesh_file), 1, 0);
  serac::StateManager::setMesh(std::move(pmesh));

  // Initialize the second order thermal solver on the parallel mesh
  auto therm_solver =
      std::make_unique<ThermalConduction>(2, ThermalConduction::defaultQuasistaticOptions(), "first_thermal");

  // Initialize the temperature boundary condition
  auto u_0 = std::make_shared<mfem::FunctionCoefficient>([](const mfem::Vector& x) { return x.Norml2(); });

  std::set<int> temp_bdr = {1};
  // Set the temperature BC in the thermal solver
  therm_solver->setTemperatureBCs(temp_bdr, u_0);

  // Set the conductivity of the thermal operator
  auto kappa = std::make_unique<mfem::ConstantCoefficient>(0.5);
  therm_solver->setConductivity(std::move(kappa));

  // Complete the setup without allocating the mass matrices and dynamic
  // operator
  therm_solver->completeSetup();

  // Destruct the old thermal solver and build a new one
  therm_solver.reset(new ThermalConduction(1, ThermalConduction::defaultQuasistaticOptions(), "second_thermal"));

  // Destruct the second thermal solver and leave the pointer empty
  therm_solver.reset(nullptr);
}

}  // namespace serac

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}

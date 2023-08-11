// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/heat_transfer.hpp"

#include <functional>
#include <set>
#include <string>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/thermal_material.hpp"
#include "serac/serac_config.hpp"

namespace serac {

TEST(HeatTransferDynamic, HeatTransferD)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "thermal_dynamic_solve");

  constexpr int dim = 2;
  constexpr int p   = 1;

  std::string filename = std::string(SERAC_REPO_DIR) + "/data/meshes/star.mesh";
  auto        mesh     = mesh::refineAndDistribute(buildMeshFromFile(filename), 2);
  serac::StateManager::setMesh(std::move(mesh));

  // Construct a heat transfer solver
  NonlinearSolverOptions nonlinear_opts{.relative_tol = 5.0e-13, .absolute_tol = 5.0e-13};

  TimesteppingOptions dyn_opts{.timestepper        = TimestepMethod::BackwardEuler,
                               .enforcement_method = DirichletEnforcementMethod::DirectControl};

  HeatTransfer<p, dim> thermal(nonlinear_opts, heat_transfer::direct_linear_options, dyn_opts, "thermal");

  heat_transfer::LinearIsotropicConductor mat(1.0, 1.0, 1.0);
  thermal.setMaterial(mat);

  // initial conditions
  thermal.setTemperature([](const mfem::Vector&, double) { return 0.0; });

  thermal.setTemperatureBCs({1}, [](auto, auto) { return 0.0; });

  thermal.setSource([](auto /* X */, auto /* time */, auto /* u */, auto /* du_dx */) { return 1.0; });

  // Finalize the data structures
  thermal.completeSetup();

  // Integrate in time
  double dt            = 0.1;
  int    num_timesteps = 3;
  thermal.outputState();
  for (int i = 0; i < num_timesteps; i++) {
    thermal.advanceTimestep(dt);
    thermal.outputState();
  }

  // Make a stand-in pseudo-load. This corresponds to the time integral of the L_1 norm of the temperature
  serac::FiniteElementDual adjoint_load(thermal.temperature().space(), "adjoint_load");
  adjoint_load = 1.0;
  for (int i = 0; i < num_timesteps; ++i) {
    thermal.reverseAdjointTimestep(dt, {{"temperature", adjoint_load}});
  }

  thermal.outputState();
}

}  // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}

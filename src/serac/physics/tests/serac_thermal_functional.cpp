// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/thermal_conduction_functional.hpp"

#include <fstream>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/serac_config.hpp"

namespace serac {

template <int p, int dim>
void functional_test()
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement   = 1;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "serac", "dynamic_solve");

  static_assert(dim == 2 || dim == 3, "Dimension must be 2 or 3 for thermal functional test");

  if constexpr (dim == 2) {
    std::string meshfile2D = SERAC_REPO_DIR "/data/meshes/star.mesh";
    auto mesh2D = mesh::refineAndDistribute(buildMeshFromFile(meshfile2D), serial_refinement, parallel_refinement);

    serac::StateManager::setMesh(std::move(mesh2D));
  }

  if constexpr (dim == 3) {
    std::string meshfile3D = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";
    auto mesh3D = mesh::refineAndDistribute(buildMeshFromFile(meshfile3D), serial_refinement, parallel_refinement);

    serac::StateManager::setMesh(std::move(mesh3D));
  }

  // define a boundary attribute set
  std::set<int> ess_bdr = {1};

  ThermalConductionFunctional<p, dim> thermal_solver(ThermalConductionFunctional<p, dim>::defaultQuasistaticOptions(),
                                                     "thermal_functional");

  ConstantIsotropicMaterial mat{.rho = 1.0, .cp = 1.0, .kappa = 1.0};
  ScalarFunction            one{1.0};

  thermal_solver.setTemperatureBCs(ess_bdr, one);
  thermal_solver.setMaterial(mat);
  thermal_solver.setSource(one);
  thermal_solver.setTemperature(one);

  thermal_solver.completeSetup();

  double dt = 1.0;
  thermal_solver.advanceTimestep(dt);
}

TEST(thermal_functional, 2D_linear) { functional_test<1, 2>(); }
TEST(thermal_functional, 2D_quad) { functional_test<2, 2>(); }

TEST(thermal_functional, 3D_linear) { functional_test<1, 3>(); }
TEST(thermal_functional, 3D_quad) { functional_test<2, 3>(); }

}  // namespace serac

//------------------------------------------------------------------------------
#include "axom/slic/core/SimpleLogger.hpp"

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}

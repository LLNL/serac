// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/thermal_solid_functional.hpp"
#include "serac/physics/materials/thermal_functional_material.hpp"
#include "serac/physics/materials/solid_functional_material.hpp"
#include "serac/physics/state/state_manager.hpp"

namespace serac {

template <int p, int dim>
void functional_test_static(double expected_norm)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement   = 1;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "thermal_solid_functional_static_solve");

  static_assert(dim == 2 || dim == 3, "Dimension must be 2 or 3 for thermal functional test");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename =
      (dim == 2) ? SERAC_REPO_DIR "/data/meshes/star.mesh" : SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  // Define a boundary attribute set
  std::set<int> ess_bdr = {1};

  // define the thermal solver configurations
  auto thermal_options         = Thermal::defaultQuasistaticOptions();
  auto solid_mechanics_options = solid_mechanics::default_static_options;

  // Construct a functional-based thermal-solid solver
  // BT 04/27/2022 This can't be instantiated yet.
  // The material model needs to be implemented before this
  // module can be used.
  ThermalSolidFunctional<p, dim> thermal_solid_solver(thermal_options, solid_mechanics_options,
                                                      GeometricNonlinearities::On, FinalMeshOption::Deformed,
                                                      "thermal_solid_functional");

  double u = 0.0;
  EXPECT_NEAR(u, expected_norm, 1.0e-6);
}

TEST(ThermalSolidFunctional, Construct) { functional_test_static<1, 2>(0.0); }

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

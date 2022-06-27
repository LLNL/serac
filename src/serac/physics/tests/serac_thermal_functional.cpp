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
#include "serac/physics/thermal_conduction_functional.hpp"
#include "serac/physics/materials/thermal_functional_material.hpp"
#include "serac/physics/materials/parameterized_thermal_functional_material.hpp"
#include "serac/physics/state/state_manager.hpp"

namespace serac {

template <int p, int dim>
void functional_test_static(double expected_temp_norm)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement   = 1;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "thermal_functional_static_solve");

  static_assert(dim == 2 || dim == 3, "Dimension must be 2 or 3 for thermal functional test");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename =
      (dim == 2) ? SERAC_REPO_DIR "/data/meshes/star.mesh" : SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  // Define a boundary attribute set
  std::set<int> ess_bdr = {1};

  // Construct a functional-based thermal conduction solver
  ThermalConductionFunctional<p, dim> thermal_solver(Thermal::defaultQuasistaticOptions(), "thermal_functional");

  tensor<double, dim, dim> cond;

  // Define an anisotropic conductor material model
  if constexpr (dim == 2) {
    cond = {{{5.0, 0.01}, {0.01, 1.0}}};
  }

  if constexpr (dim == 3) {
    cond = {{{1.5, 0.01, 0.0}, {0.01, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
  }

  Thermal::LinearConductor<dim> mat(1.0, 1.0, cond);
  thermal_solver.setMaterial(mat);

  // Define the function for the initial temperature and boundary condition
  auto one = [](const mfem::Vector&, double) -> double { return 1.0; };

  // Set the initial temperature and boundary condition
  thermal_solver.setTemperatureBCs(ess_bdr, one);
  thermal_solver.setTemperature(one);

  // Define a constant source term
  Thermal::ConstantSource source{1.0};
  thermal_solver.setSource(source);

  // Set the flux term to zero for testing code paths
  Thermal::ConstantFlux flux_bc{0.0};
  thermal_solver.setFluxBCs(flux_bc);

  // Finalize the data structures
  thermal_solver.completeSetup();

  // Perform the quasi-static solve
  double dt = 1.0;
  thermal_solver.advanceTimestep(dt);

  // Output the sidre-based plot files
  thermal_solver.outputState();

  // Check the final temperature norm
  EXPECT_NEAR(expected_temp_norm, norm(thermal_solver.temperature()), 1.0e-6);
}

template <int p, int dim>
void functional_test_dynamic(double expected_temp_norm)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement   = 1;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "thermal_functional_dynamic_solve");

  static_assert(dim == 2 || dim == 3, "Dimension must be 2 or 3 for thermal functional test");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename =
      (dim == 2) ? SERAC_REPO_DIR "/data/meshes/star.mesh" : SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  // Define a boundary attribute set
  std::set<int> ess_bdr = {1};

  // Construct a functional-based thermal conduction solver
  ThermalConductionFunctional<p, dim> thermal_solver(Thermal::defaultDynamicOptions(), "thermal_functional");

  // Define an isotropic conductor material model
  Thermal::LinearIsotropicConductor mat(0.5, 0.5, 0.5);

  thermal_solver.setMaterial(mat);

  // Define the function for the initial temperature and boundary condition
  auto initial_temp = [](const mfem::Vector& x, double) -> double {
    if (std::sqrt(x[0] * x[0] + x[1] * x[1]) < 0.5) {
      return 2.0;
    }
    return 1.0;
  };

  // Set the initial temperature and boundary condition
  thermal_solver.setTemperatureBCs(ess_bdr, initial_temp);
  thermal_solver.setTemperature(initial_temp);

  // Set the flux term to zero for testing code paths
  Thermal::ConstantFlux flux_bc{0.0};
  thermal_solver.setFluxBCs(flux_bc);

  // Finalize the data structures
  thermal_solver.completeSetup();

  // Perform the time stepping
  double dt = 1.0;

  for (int i = 0; i < 5; ++i) {
    thermal_solver.outputState();
    thermal_solver.advanceTimestep(dt);
  }

  // Output the sidre-based plot files
  thermal_solver.outputState();

  // Check the final temperature norm
  EXPECT_NEAR(expected_temp_norm, norm(thermal_solver.temperature()), 1.0e-6);
}

TEST(ThermalFunctional, 2DLinearStatic) { functional_test_static<1, 2>(2.2909240); }
TEST(ThermalFunctional, 2DQuadStatic) { functional_test_static<2, 2>(2.29424403); }
TEST(ThermalFunctional, 3DLinearStatic) { functional_test_static<1, 3>(46.6285642); }
TEST(ThermalFunctional, 3DQuadStatic) { functional_test_static<2, 3>(46.6648538); }

TEST(ThermalFunctional, 2DLinearDynamic) { functional_test_dynamic<1, 2>(2.18066491); }
TEST(ThermalFunctional, 2DQuadDynamic) { functional_test_dynamic<2, 2>(2.1806651); }
TEST(ThermalFunctional, 3DLinearDynamic) { functional_test_dynamic<1, 3>(3.1447306); }
TEST(ThermalFunctional, 3DQuadDynamic) { functional_test_dynamic<2, 3>(3.36129252); }

TEST(ThermalFunctional, ParameterizedMaterial)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement   = 1;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "thermal_functional_parameterized_sensitivities");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/star.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  constexpr int p   = 1;
  constexpr int dim = 2;

  // Define a boundary attribute set
  std::set<int> ess_bdr = {1};

  // Construct and initialized the user-defined conductivity to be used as a differentiable parameter in
  // the thermal conduction physics module.
  FiniteElementState user_defined_conductivity(
      StateManager::newState(FiniteElementState::Options{.order = 1, .name = "parameterized_conductivity"}));

  user_defined_conductivity = 1.0;

  // We must know the index of the parameter finite element state in our parameter pack to take sensitivities.
  // As we only have one parameter in this example, the index is zero.
  constexpr int conductivity_parameter_index = 0;

  // Construct a functional-based thermal conduction solver
  //
  // Note that we now include an extra template parameter indicating the finite element space for the parameterized
  // field, in this case the thermal conductivity. We also pass an array of finite element states for each of the
  // requested parameterized fields.
  ThermalConductionFunctional<p, dim, H1<1>> thermal_solver(Thermal::defaultQuasistaticOptions(), "thermal_functional",
                                                            {user_defined_conductivity});

  // Construct a potentially user-defined parameterized material and send it to the thermal module
  Thermal::ParameterizedLinearIsotropicConductor mat;
  thermal_solver.setMaterial(mat);

  // Define the function for the initial temperature and boundary condition
  auto bdr_temp = [](const mfem::Vector& x, double) -> double {
    if (x[0] < 0.5 || x[1] < 0.5) {
      return 1.0;
    }
    return 0.0;
  };

  // Set the initial temperature and boundary condition
  thermal_solver.setTemperatureBCs(ess_bdr, bdr_temp);
  thermal_solver.setTemperature(bdr_temp);

  // Define a constant source term
  Thermal::ConstantSource source{1.0};
  thermal_solver.setSource(source);

  // Set the flux term to zero for testing code paths
  Thermal::ConstantFlux flux_bc{0.0};
  thermal_solver.setFluxBCs(flux_bc);

  // Finalize the data structures
  thermal_solver.completeSetup();

  // Perform the quasi-static solve
  double dt = 1.0;
  thermal_solver.advanceTimestep(dt);

  // Output the sidre-based plot files
  thermal_solver.outputState();

  // Construct a dummy adjoint load (this would come from a QOI downstream).
  // This adjoint load is equivalent to a discrete L1 norm on the temperature.
  FiniteElementDual adjoint_load(
      StateManager::newDual(FiniteElementState::Options{.order = 1, .name = "adjoint_load"}));

  adjoint_load = 1.0;

  // Solve the adjoint problem
  thermal_solver.solveAdjoint(adjoint_load);

  // Compute the sensitivity (d QOI/ d state * d state/d parameter) given the current adjoint solution
  auto& sensitivity = thermal_solver.computeSensitivity<conductivity_parameter_index>();

  EXPECT_NEAR(1.6540980, mfem::ParNormlp(sensitivity, 2, MPI_COMM_WORLD), 1.0e-6);
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

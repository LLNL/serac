// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/heat_transfer.hpp"
#include "serac/physics/materials/thermal_material.hpp"
#include "serac/physics/materials/parameterized_thermal_material.hpp"

#include <fstream>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"

namespace serac {

TEST(HeatTransfer, MoveShape)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement   = 2;
  int parallel_refinement = 0;

  constexpr int p   = 1;
  constexpr int dim = 2;

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/square.mesh";

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "thermal_shape_solve");

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  mfem::Vector shape_temperature;
  mfem::Vector pure_temperature;

  // Define a boundary attribute set
  std::set<int> ess_bdr = {1};

  // Use a direct solver for the Jacobian solve
  SolverOptions options = {DirectSolverOptions{}, Thermal::defaultNonlinearOptions()};

  // Use tight tolerances as this is a machine precision test
  options.nonlinear.abs_tol = 1.0e-14;
  options.nonlinear.rel_tol = 1.0e-14;

  // Define an anisotropic conductor material model
  tensor<double, 2, 2>          cond{{{5.0, 0.4}, {0.4, 1.0}}};
  Thermal::LinearConductor<dim> mat(1.0, 1.0, cond);

  Thermal::ConstantSource source{1.0};

  double shape_factor_1 = 200.0;
  double shape_factor_2 = 0.0;
  // Project a non-affine transformation
  mfem::VectorFunctionCoefficient shape_coef(
      2, [shape_factor_1, shape_factor_2](const mfem::Vector& x, mfem::Vector& shape) {
        shape[0] = x[1] * shape_factor_1;
        shape[1] = x[0] * shape_factor_2;
      });

  // Define the function for the initial temperature and boundary condition
  auto zero = [](const mfem::Vector&, double) -> double { return 0.0; };

  {
    // Construct a functional-based thermal solver including references to the shape displacement field.
    HeatTransfer<p, dim> thermal_solver(options, "thermal_shape", ShapeDisplacement::On);

    // Set the initial temperature and boundary condition
    thermal_solver.setTemperatureBCs(ess_bdr, zero);
    thermal_solver.setTemperature(zero);

    thermal_solver.shapeDisplacement().project(shape_coef);

    thermal_solver.setMaterial(mat);

    thermal_solver.setSource(source);

    // Finalize the data structures
    thermal_solver.completeSetup();

    // Perform the quasi-static solve
    double dt = 1.0;
    thermal_solver.advanceTimestep(dt);

    thermal_solver.outputState();

    shape_temperature = thermal_solver.temperature().gridFunction();
  }

  axom::sidre::DataStore new_datastore;
  StateManager::reset();
  serac::StateManager::initialize(new_datastore, "thermal_pure_solve");

  auto new_mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(new_mesh));

  {
    // Construct and initialized the user-defined shape displacement to offset the computational mesh
    FiniteElementState user_defined_shape_displacement(StateManager::newState(
        FiniteElementState::Options{.order = p, .vector_dim = dim, .name = "parameterized_shape"}));

    user_defined_shape_displacement.project(shape_coef);

    // Delete the pre-computed geometry factors as we are mutating the mesh
    StateManager::mesh().DeleteGeometricFactors();
    auto* mesh_nodes = StateManager::mesh().GetNodes();
    *mesh_nodes += user_defined_shape_displacement.gridFunction();

    // Construct a functional-based thermal solver including references to the shape displacement field.
    HeatTransfer<p, dim> thermal_solver_no_shape(options, "thermal_pure", ShapeDisplacement::Off);

    // Set the initial temperature and boundary condition
    thermal_solver_no_shape.setTemperatureBCs(ess_bdr, zero);
    thermal_solver_no_shape.setTemperature(zero);

    thermal_solver_no_shape.setMaterial(mat);

    thermal_solver_no_shape.setSource(source);

    // Finalize the data structures
    thermal_solver_no_shape.completeSetup();

    // Perform the quasi-static solve
    double dt = 1.0;
    thermal_solver_no_shape.advanceTimestep(dt);

    thermal_solver_no_shape.outputState();

    pure_temperature = thermal_solver_no_shape.temperature().gridFunction();
  }

  double error          = pure_temperature.DistanceTo(shape_temperature.GetData());
  double relative_error = error / pure_temperature.Norml2();
  EXPECT_LT(relative_error, 1.0e-14);
}

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

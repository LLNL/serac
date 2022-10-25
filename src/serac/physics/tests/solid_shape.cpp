// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_mechanics.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/physics/materials/parameterized_solid_material.hpp"

#include <fstream>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"

namespace serac {

void shape_test(GeometricNonlinearities geo_nonlin)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement   = 0;
  int parallel_refinement = 0;

  constexpr int p   = 1;
  constexpr int dim = 2;

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/beam-quad.mesh";

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_functional_shape_solve");

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  mfem::Vector shape_displacement;
  mfem::Vector pure_displacement;

  // Define a boundary attribute set
  std::set<int> ess_bdr = {1};

  // Use a krylov solver for the Jacobian solve
  SolverOptions options = {solid_mechanics::default_linear_options, solid_mechanics::default_nonlinear_options};

  // Use tight tolerances as this is a machine precision test
  get<IterativeSolverOptions>(options.linear).rel_tol = 1.0e-15;
  get<IterativeSolverOptions>(options.linear).abs_tol = 1.0e-15;

  options.nonlinear.abs_tol = 1.0e-15;
  options.nonlinear.rel_tol = 1.0e-15;
  options.nonlinear.max_iter = 5;

  solid_mechanics::LinearIsotropic mat{1.0, 1.0, 1.0};

  double shape_factor = 2.0;

  // Define the function for the initial displacement and boundary condition
  auto bc = [](const mfem::Vector& x, mfem::Vector& bc_vec) -> void {
    bc_vec[0] = 0.0;
    bc_vec[1] = x[0] * 0.1;
  };

  // Define the function for the initial displacement and boundary condition
  auto bc_pure = [shape_factor](const mfem::Vector& x, mfem::Vector& bc_vec) -> void {
    bc_vec[0] = 0.0;
    bc_vec[1] = (x[0] * 0.1) / (shape_factor + 1.0);
  };

  // Construct and apply a uniform body load
  tensor<double, dim> constant_force;

  constant_force[0] = 0.0e-3;
  constant_force[1] = 1.0e-4;

  if (dim == 3) {
    constant_force[2] = 0.0;
  }

  solid_mechanics::ConstantBodyForce<dim> force{constant_force};

  // Project a non-affine transformation with an affine transformation on the boundary
  mfem::VectorFunctionCoefficient shape_coef(2, [shape_factor](const mfem::Vector& x, mfem::Vector& shape) {
    shape[0] = x[0] * shape_factor;
    shape[1] = 0.0;
  });

  {
    // Construct and initialized the user-defined shape velocity to offset the computational mesh
    FiniteElementState user_defined_shape_displacement(StateManager::newState(
        FiniteElementState::Options{.order = p, .vector_dim = dim, .name = "parameterized_shape"}));

    user_defined_shape_displacement.project(shape_coef);

    // Construct a functional-based solid mechanics solver including references to the shape velocity field.
    SolidMechanics<p, dim> solid_solver(options, geo_nonlin, "solid_functional", ShapeDisplacement::On);

    // Set the initial displacement and boundary condition
    solid_solver.setDisplacementBCs(ess_bdr, bc);
    solid_solver.setDisplacement(bc);

    solid_solver.shapeDisplacement() = user_defined_shape_displacement;

    solid_solver.setMaterial(mat);

    solid_solver.addBodyForce(force);

    // Finalize the data structures
    solid_solver.completeSetup();

    // Perform the quasi-static solve
    double dt = 1.0;
    solid_solver.advanceTimestep(dt);

    shape_displacement = solid_solver.displacement().gridFunction();
  }

  axom::sidre::DataStore new_datastore;
  StateManager::reset();
  serac::StateManager::initialize(new_datastore, "solid_functional_pure_solve");

  auto new_mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(new_mesh));

  {
    // Construct and initialized the user-defined shape velocity to offset the computational mesh
    FiniteElementState user_defined_shape_displacement(StateManager::newState(
        FiniteElementState::Options{.order = p, .vector_dim = dim, .name = "parameterized_shape"}));

    user_defined_shape_displacement.project(shape_coef);

    // Delete the pre-computed geometry factors as we are mutating the mesh
    StateManager::mesh().DeleteGeometricFactors();
    auto* mesh_nodes = StateManager::mesh().GetNodes();
    *mesh_nodes += user_defined_shape_displacement.gridFunction();

    // Construct a functional-based solid mechanics solver including references to the shape velocity field.
    SolidMechanics<p, dim> solid_solver_no_shape(options, geo_nonlin, "solid_functional");

    mfem::VisItDataCollection visit_dc("pure_version", const_cast<mfem::ParMesh*>(&solid_solver_no_shape.mesh()));
    visit_dc.RegisterField("displacement", &solid_solver_no_shape.displacement().gridFunction());
    visit_dc.Save();

    // Set the initial displacement and boundary condition
    solid_solver_no_shape.setDisplacementBCs(ess_bdr, bc_pure);
    solid_solver_no_shape.setDisplacement(bc_pure);

    solid_solver_no_shape.setMaterial(mat);

    solid_solver_no_shape.addBodyForce(force);

    // Finalize the data structures
    solid_solver_no_shape.completeSetup();

    // Perform the quasi-static solve
    double dt = 1.0;
    solid_solver_no_shape.advanceTimestep(dt);

    pure_displacement = solid_solver_no_shape.displacement().gridFunction();
    visit_dc.SetCycle(1);
    visit_dc.Save();
  }

  double error          = pure_displacement.DistanceTo(shape_displacement.GetData());
  double relative_error = error / pure_displacement.Norml2();
  EXPECT_LT(relative_error, 1.0e-14);
}

TEST(SolidMechanics, MoveShapeLinear) { shape_test(GeometricNonlinearities::Off); }
TEST(SolidMechanics, MoveShapeNonlinear) { shape_test(GeometricNonlinearities::On); }

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

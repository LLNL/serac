// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_functional.hpp"
#include "serac/physics/materials/solid_functional_material.hpp"
#include "serac/physics/materials/parameterized_solid_functional_material.hpp"

#include <fstream>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"

namespace serac {

TEST(solid_functional_finite_diff, finite_difference)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement   = 1;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_functional_parameterized_sensitivities");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/beam-quad.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  constexpr int p   = 1;
  constexpr int dim = 2;

  // Define a boundary attribute set
  std::set<int> ess_bdr = {1};

  // define the solver configurations
  const IterativeSolverOptions default_linear_options = {.rel_tol     = 1.0e-8,
                                                         .abs_tol     = 1.0e-14,
                                                         .print_level = 0,
                                                         .max_iter    = 500,
                                                         .lin_solver  = LinearSolver::GMRES,
                                                         .prec        = HypreBoomerAMGPrec{}};

  const NonlinearSolverOptions default_nonlinear_options = {
      .rel_tol = 1.0e-6, .abs_tol = 1.0e-12, .max_iter = 10, .print_level = 1};

  const typename solid_util::SolverOptions default_static = {default_linear_options, default_nonlinear_options};

  // Construct and initialized the user-defined moduli to be used as a differentiable parameter in
  // the solid physics module.
  FiniteElementState user_defined_shear_modulus(
      StateManager::newState(FiniteElementState::Options{.order = 1, .name = "parameterized_shear"}));

  double shear_modulus_value = 1.0;

  user_defined_shear_modulus = shear_modulus_value;

  FiniteElementState user_defined_bulk_modulus(
      StateManager::newState(FiniteElementState::Options{.order = 1, .name = "parameterized_bulk"}));

  double bulk_modulus_value = 1.0;

  user_defined_bulk_modulus = bulk_modulus_value;

  // Construct a functional-based solid solver
  SolidFunctional<p, dim, H1<1>, H1<1>> solid_solver(default_static, GeometricNonlinearities::On,
                                                     FinalMeshOption::Reference, "solid_functional",
                                                     {user_defined_bulk_modulus, user_defined_shear_modulus});

  // We must know the index of the parameter finite element state in our parameter pack to take sensitivities.
  // As we only have one parameter in this example, the index is zero.
  constexpr int bulk_parameter_index = 0;

  solid_util::ParameterizedNeoHookeanSolid<dim> mat(1.0, 0.0, 0.0);
  solid_solver.setMaterial(mat);

  // Define the function for the initial displacement and boundary condition
  auto bc = [](const mfem::Vector&, mfem::Vector& bc_vec) -> void { bc_vec = 0.0; };

  // Set the initial displacement and boundary condition
  solid_solver.setDisplacementBCs(ess_bdr, bc);
  solid_solver.setDisplacement(bc);

  tensor<double, dim> constant_force;

  constant_force[0] = 0.0;
  constant_force[1] = 1.0e-3;

  if (dim == 3) {
    constant_force[2] = 0.0;
  }

  solid_util::ConstantBodyForce<dim> force{constant_force};
  solid_solver.addBodyForce(force);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  double dt = 1.0;
  solid_solver.advanceTimestep(dt);

  // Output the sidre-based plot files
  solid_solver.outputState();

  // Make up an adjoint load which can also be viewed as a
  // sensitivity of some qoi with respect to displacement
  mfem::ParLinearForm adjoint_load_form(&solid_solver.displacement().space());
  adjoint_load_form = 1.0;

  // Construct a dummy adjoint load (this would come from a QOI downstream).
  // This adjoint load is equivalent to a discrete L1 norm on the displacement.
  serac::FiniteElementDual              adjoint_load(*mesh, solid_solver.displacement().space(), "adjoint_load");
  std::unique_ptr<mfem::HypreParVector> assembled_vector(adjoint_load_form.ParallelAssemble());
  adjoint_load.trueVec() = *assembled_vector;
  adjoint_load.distributeSharedDofs();

  // Solve the adjoint problem
  solid_solver.solveAdjoint(adjoint_load);

  // Compute the sensitivity (d QOI/ d state * d state/d parameter) given the current adjoint solution
  [[maybe_unused]] auto& sensitivity = solid_solver.computeSensitivity<bulk_parameter_index>();

  // Perform finite difference on each bulk modulus value
  // to check if computed qoi sensitivity is consistent
  // with finite difference on the displacement
  double eps = 1.0e-6;
  for (int i = 0; i < user_defined_bulk_modulus.gridFunc().Size(); ++i) {
    // Perturb the bulk modulus
    user_defined_bulk_modulus.trueVec()(i) = bulk_modulus_value + eps;
    user_defined_bulk_modulus.distributeSharedDofs();

    solid_solver.advanceTimestep(dt);
    mfem::ParGridFunction displacement_plus = solid_solver.displacement().gridFunc();

    user_defined_bulk_modulus.trueVec()(i) = bulk_modulus_value - eps;
    user_defined_bulk_modulus.distributeSharedDofs();

    solid_solver.advanceTimestep(dt);
    mfem::ParGridFunction displacement_minus = solid_solver.displacement().gridFunc();

    // Reset to the original bulk modulus value
    user_defined_bulk_modulus.trueVec()(i) = bulk_modulus_value;
    user_defined_bulk_modulus.distributeSharedDofs();

    // Finite difference to compute sensitivity of displacement with respect to bulk modulus
    mfem::ParGridFunction ddisp_dbulk(&solid_solver.displacement().space());
    for (int i2 = 0; i2 < displacement_plus.Size(); ++i2) {
      ddisp_dbulk(i2) = (displacement_plus(i2) - displacement_minus(i2)) / (2.0 * eps);
    }

    // Compute numerical value of sensitivity of qoi with respect to bulk modulus
    // by taking the inner product between adjoint load and displacement sensitivity
    double dqoi_dbulk = adjoint_load_form(ddisp_dbulk);

    // See if these are similar
    SLIC_INFO(axom::fmt::format("dqoi_dbulk: {}", dqoi_dbulk));
    SLIC_INFO(axom::fmt::format("sensitivity: {}", sensitivity.trueVec()(i)));
    EXPECT_NEAR((sensitivity.trueVec()(i) - dqoi_dbulk) / std::max(dqoi_dbulk, 1.0e-2), 0.0, 5.0e-3);
  }
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

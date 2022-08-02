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
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/solid_functional.hpp"
#include "serac/physics/materials/solid_functional_material.hpp"
#include "serac/physics/materials/parameterized_solid_functional_material.hpp"

namespace serac {

template <int dim>
mfem::Mesh buildHypercubeMesh(std::array<int, dim> elementsPerDim)
{
  if constexpr (dim == 2) {
    return buildRectangleMesh(elementsPerDim[0], elementsPerDim[1]);
  } else if constexpr (dim == 3) {
    return buildCuboidMesh(elementsPerDim[0], elementsPerDim[1], elementsPerDim[2]);
  }
}

template <int dim>
double patch_test(std::function<void(const mfem::Vector&, mfem::Vector&)> exact_displacement_function)
{
  constexpr int p = 1;
  
  //auto body_force_function = [](auto, auto, auto, auto){ return tensor<double, dim>{}; };

  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement   = 0;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_functional_static_solve");

  static_assert(dim == 2 || dim == 3, "Dimension must be 2 or 3 for solid functional test");

  // Construct the appropriate dimension mesh and give it to the data store
  std::array<int, dim> elements_per_dim{3, 4};
  if constexpr (dim == 3) {
    elements_per_dim[2] = 2;
  }
  auto mesh = mesh::refineAndDistribute(buildHypercubeMesh<dim>(elements_per_dim),
                                        serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  // define the solver configurations
  const IterativeSolverOptions default_linear_options = {.rel_tol     = 1.0e-10,
                                                         .abs_tol     = 1.0e-11,
                                                         .print_level = 0,
                                                         .max_iter    = 20,
                                                         .lin_solver  = LinearSolver::GMRES,
                                                         .prec        = HypreBoomerAMGPrec{}};

  const NonlinearSolverOptions default_nonlinear_options = {
      .rel_tol = 1.0e-12, .abs_tol = 1.0e-15, .max_iter = 10, .print_level = 1};

  const typename solid_util::SolverOptions default_static = {default_linear_options, default_nonlinear_options};

  // Construct a functional-based solid mechanics solver
  SolidFunctional<p, dim> solid_solver(default_static, GeometricNonlinearities::On, FinalMeshOption::Reference,
                                       "solid_functional");

  double density = 1.0;
  double shear_modulus = 1.0;
  double bulk_modulus = 1.0;
  solid_util::NeoHookeanSolid<dim> mat(density, shear_modulus, bulk_modulus);
  solid_solver.setMaterial(mat);

  // Set the initial displacement
  //solid_solver.setDisplacement([](const mfem::Vector&, mfem::Vector& u) { u = 0.0; });
  
  // Define a boundary attribute set
  std::set<int> essential_boundaries;
  if constexpr (dim == 2) {
    essential_boundaries = {1, 2, 3, 4};
  } else {
    essential_boundaries = {1, 2, 3, 4, 5, 6};
  }
  
  // displacement boundary condition
  solid_solver.setDisplacementBCs(essential_boundaries, exact_displacement_function);

  // traction
  // tensor<double, dim> unused{};
  // auto H = make_tensor<dim, dim>([A](int i, int j) { return A(i,j); }); 
  // auto response = mat(unused, unused, H);
  // tensor<double, dim, dim> tau = response.stress;
  // std::cout << "stress\n" << tau << std::endl;
  // solid_solver.setTractionBCs([=](auto, auto n, auto) { return dot(tau, n); });

  //solid_solver.addBodyForce(force);
  
  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  double dt = 1.0;
  solid_solver.advanceTimestep(dt);

  // Output the sidre-based and paraview plot files
  solid_solver.outputState("paraview_output");

  // Compute norm of error
  mfem::VectorFunctionCoefficient exact_solution_coef(dim, exact_displacement_function);
  double uNorm = solid_solver.displacement().gridFunction().Norml2();
  std::cout << "||u|| = " << uNorm << std::endl;
  return solid_solver.displacement().gridFunction().ComputeL2Error(exact_solution_coef);
}


template <int p, int dim>
void functional_solid_test_static(double expected_disp_norm)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement   = 0;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_functional_static_solve");

  static_assert(dim == 2 || dim == 3, "Dimension must be 2 or 3 for solid functional test");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename =
      (dim == 2) ? SERAC_REPO_DIR "/data/meshes/beam-quad.mesh" : SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  // Define a boundary attribute set
  std::set<int> ess_bdr = {1};

  // define the solver configurations
  const DirectSolverOptions direct_options{.print_level = 1};

  const NonlinearSolverOptions default_nonlinear_options = {
      .rel_tol = 1.0e-4, .abs_tol = 1.0e-8, .max_iter = 10, .print_level = 1};

  const typename solid_util::SolverOptions default_static = {direct_options, default_nonlinear_options};

  // Construct a functional-based solid mechanics solver
  SolidFunctional<p, dim> solid_solver(default_static, GeometricNonlinearities::On, FinalMeshOption::Reference,
                                       "solid_functional");

  solid_util::NeoHookeanSolid<dim> mat(1.0, 1.0, 1.0);
  solid_solver.setMaterial(mat);

  // Define the function for the initial displacement and boundary condition
  auto bc = [](const mfem::Vector&, mfem::Vector& bc_vec) -> void { bc_vec = 0.0; };

  // Set the initial displacement and boundary condition
  solid_solver.setDisplacementBCs(ess_bdr, bc);
  solid_solver.setDisplacement(bc);

  tensor<double, dim> constant_force;

  constant_force[0] = 0.0;
  constant_force[1] = 5.0e-4;

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

  // Output the sidre-based and paraview plot files
  solid_solver.outputState("paraview_output");

  // Check the final displacement norm
  EXPECT_NEAR(expected_disp_norm, norm(solid_solver.displacement()), 1.0e-6);
}

template <int p, int dim>
void functional_solid_test_dynamic(double expected_disp_norm)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement   = 0;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_functional_dynamic_solve");

  static_assert(dim == 2 || dim == 3, "Dimension must be 2 or 3 for solid functional test");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename =
      (dim == 2) ? SERAC_REPO_DIR "/data/meshes/beam-quad.mesh" : SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  // Define a boundary attribute set
  std::set<int> ess_bdr = {1};

  // define the solver configurations
  const IterativeSolverOptions default_linear_options = {.rel_tol     = 1.0e-6,
                                                         .abs_tol     = 1.0e-10,
                                                         .print_level = 0,
                                                         .max_iter    = 500,
                                                         .lin_solver  = LinearSolver::GMRES,
                                                         .prec        = HypreBoomerAMGPrec{}};

  const NonlinearSolverOptions default_nonlinear_options = {
      .rel_tol = 1.0e-4, .abs_tol = 1.0e-8, .max_iter = 10, .print_level = 1};

  const typename solid_util::TimesteppingOptions default_timestep = {TimestepMethod::AverageAcceleration,
                                                                     DirichletEnforcementMethod::RateControl};

  const typename solid_util::SolverOptions default_dynamic = {default_linear_options, default_nonlinear_options,
                                                              default_timestep};

  // Construct a functional-based solid mechanics solver
  SolidFunctional<p, dim> solid_solver(default_dynamic, GeometricNonlinearities::Off, FinalMeshOption::Reference,
                                       "solid_functional_dynamic");

  solid_util::LinearIsotropicSolid<dim> mat(1.0, 1.0, 1.0);
  solid_solver.setMaterial(mat);

  // Define the function for the initial displacement and boundary condition
  auto bc = [](const mfem::Vector&, mfem::Vector& bc_vec) -> void { bc_vec = 0.0; };

  // Set the initial displacement and boundary condition
  solid_solver.setDisplacementBCs(ess_bdr, bc);
  solid_solver.setDisplacement(bc);

  tensor<double, dim> constant_force;

  constant_force[0] = 0.0;
  constant_force[1] = 5.0e-1;

  if (dim == 3) {
    constant_force[2] = 0.0;
  }

  solid_util::ConstantBodyForce<dim> force{constant_force};
  solid_solver.addBodyForce(force);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  double dt = 0.5;

  for (int i = 0; i < 3; ++i) {
    solid_solver.advanceTimestep(dt);
    solid_solver.outputState();
  }

  // Check the final displacement norm
  EXPECT_NEAR(expected_disp_norm, norm(solid_solver.displacement()), 1.0e-6);
}

enum class TestType
{
  Pressure,
  Traction
};

template <int p, int dim>
void functional_solid_test_boundary(double expected_disp_norm, TestType test_mode)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement   = 1;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_functional_static_solve");

  static_assert(dim == 2 || dim == 3, "Dimension must be 2 or 3 for solid functional test");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename =
      (dim == 2) ? SERAC_REPO_DIR "/data/meshes/beam-quad.mesh" : SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  // Define a boundary attribute set
  std::set<int> ess_bdr = {1};

  // define the solver configurations
  const IterativeSolverOptions default_linear_options = {.rel_tol     = 1.0e-6,
                                                         .abs_tol     = 1.0e-10,
                                                         .print_level = 0,
                                                         .max_iter    = 500,
                                                         .lin_solver  = LinearSolver::GMRES,
                                                         .prec        = HypreBoomerAMGPrec{}};

  const NonlinearSolverOptions default_nonlinear_options = {
      .rel_tol = 1.0e-4, .abs_tol = 1.0e-8, .max_iter = 10, .print_level = 1};

  const typename solid_util::SolverOptions default_static = {default_linear_options, default_nonlinear_options};

  // Construct a functional-based solid mechanics solver
  SolidFunctional<p, dim> solid_solver(default_static, GeometricNonlinearities::Off, FinalMeshOption::Reference,
                                       "solid_functional");

  solid_util::LinearIsotropicSolid<dim> mat(1.0, 1.0, 1.0);
  solid_solver.setMaterial(mat);

  // Define the function for the initial displacement and boundary condition
  auto bc = [](const mfem::Vector&, mfem::Vector& bc_vec) -> void { bc_vec = 0.0; };

  // Set the initial displacement and boundary condition
  solid_solver.setDisplacementBCs(ess_bdr, bc);
  solid_solver.setDisplacement(bc);

  if (test_mode == TestType::Pressure) {
    solid_util::PressureFunction<dim> pressure{[](const tensor<double, dim>& x, const double) {
      if (x[0] > 7.5) {
        return 1.0e-2;
      }
      return 0.0;
    }};
    solid_solver.setPressureBCs(pressure);
  } else if (test_mode == TestType::Traction) {
    solid_util::TractionFunction<dim> traction_function{
        [](const tensor<double, dim>& x, const tensor<double, dim>&, const double) {
          tensor<double, dim> traction;
          for (int i = 0; i < dim; ++i) {
            traction[i] = 0.0;
          }

          if (x[0] > 7.9) {
            traction[1] = 1.0e-4;
          }
          return traction;
        }};
    solid_solver.setTractionBCs(traction_function);
  } else {
    // Default to fail if non-implemented TestType is not implemented
    EXPECT_TRUE(false);
  }

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  double dt = 1.0;
  solid_solver.advanceTimestep(dt);

  // Output the sidre-based plot files
  solid_solver.outputState();

  // Check the final displacement norm
  EXPECT_NEAR(expected_disp_norm, norm(solid_solver.displacement()), 1.0e-6);
}

template <int p, int dim>
void functional_parameterized_solid_test(double expected_disp_norm)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement   = 0;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_functional_parameterized_solve");

  static_assert(dim == 2 || dim == 3, "Dimension must be 2 or 3 for solid functional parameterized test");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename =
      (dim == 2) ? SERAC_REPO_DIR "/data/meshes/beam-quad.mesh" : SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  // Define a boundary attribute set
  std::set<int> ess_bdr = {1};

  // define the solver configurations
  const IterativeSolverOptions default_linear_options = {.rel_tol     = 1.0e-6,
                                                         .abs_tol     = 1.0e-10,
                                                         .print_level = 0,
                                                         .max_iter    = 500,
                                                         .lin_solver  = LinearSolver::GMRES,
                                                         .prec        = HypreBoomerAMGPrec{}};

  const NonlinearSolverOptions default_nonlinear_options = {
      .rel_tol = 1.0e-4, .abs_tol = 1.0e-8, .max_iter = 10, .print_level = 1};

  const typename solid_util::SolverOptions default_static = {default_linear_options, default_nonlinear_options};

  // Construct and initialized the user-defined moduli to be used as a differentiable parameter in
  // the solid mechanics physics module.
  FiniteElementState user_defined_shear_modulus(
      StateManager::newState(FiniteElementState::Options{.order = 1, .name = "parameterized_shear"}));

  user_defined_shear_modulus = 1.0;

  FiniteElementState user_defined_bulk_modulus(
      StateManager::newState(FiniteElementState::Options{.order = 1, .name = "parameterized_bulk"}));

  user_defined_bulk_modulus = 1.0;

  // Construct a functional-based solid mechanics solver
  SolidFunctional<p, dim, H1<1>, H1<1>> solid_solver(default_static, GeometricNonlinearities::On,
                                                     FinalMeshOption::Reference, "solid_functional",
                                                     {user_defined_bulk_modulus, user_defined_shear_modulus});

  solid_util::ParameterizedNeoHookeanSolid<dim> mat(1.0, 0.0, 0.0);
  solid_solver.setMaterial(mat);

  // Define the function for the initial displacement and boundary condition
  auto bc = [](const mfem::Vector&, mfem::Vector& bc_vec) -> void { bc_vec = 0.0; };

  // Set the initial displacement and boundary condition
  solid_solver.setDisplacementBCs(ess_bdr, bc);
  solid_solver.setDisplacement(bc);

  tensor<double, dim> constant_force;

  constant_force[0] = 0.0;
  constant_force[1] = 5.0e-4;

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

  // Check the final displacement norm
  EXPECT_NEAR(expected_disp_norm, norm(solid_solver.displacement()), 1.0e-6);
}

TEST(SolidFunctional, 2DLinearStatic) { functional_solid_test_static<1, 2>(1.511052595); }
TEST(SolidFunctional, 2DQuadStatic) { functional_solid_test_static<2, 2>(2.18604855); }
TEST(SolidFunctional, 2DQuadParameterizedStatic) { functional_parameterized_solid_test<2, 2>(2.18604855); }

TEST(SolidFunctional, 3DLinearStatic) { functional_solid_test_static<1, 3>(1.37084852); }
TEST(SolidFunctional, 3DQuadStatic) { functional_solid_test_static<2, 3>(1.949532747); }

TEST(SolidFunctional, 2DLinearDynamic) { functional_solid_test_dynamic<1, 2>(1.52116682); }
TEST(SolidFunctional, 2DQuadDynamic) { functional_solid_test_dynamic<2, 2>(1.52777214); }

TEST(SolidFunctional, 3DLinearDynamic) { functional_solid_test_dynamic<1, 3>(1.520679017); }
TEST(SolidFunctional, 3DQuadDynamic) { functional_solid_test_dynamic<2, 3>(1.527009514); }

TEST(SolidFunctional, 2DLinearPressure) { functional_solid_test_boundary<1, 2>(0.065326222, TestType::Pressure); }
TEST(SolidFunctional, 2DLinearTraction) { functional_solid_test_boundary<1, 2>(0.126593590, TestType::Traction); }

template <int dim>
void affine_solution(const mfem::Vector& X, mfem::Vector u)
{
  mfem::DenseMatrix A(dim);
  mfem::Vector b(dim);
  A(0, 0) = 0.110791568544027;
  A(0, 1) = 0.230421268325901;
  A(1, 0) = 0.198344644470483;
  A(1, 1) = 0.060514559793513;
  if constexpr (dim == 3) {
    A(0, 2) = 0.15167673653354;
    A(1, 2) = 0.084137393813728;
    A(2, 0) = 0.011544253485023;
    A(2, 1) = 0.060942846497753;
    A(2, 2) = 0.186383473579596;
  }
  
  b(0) = 0.765645367640828;
  b(1) = 0.992487355850465;
  if constexpr (dim == 3) {
    b(2) = 0.162199373722092;
  }

  A.Mult(X, u);
  u += b;
  //u *= 0.01;
}

TEST(SolidFunctional, Patch2D)
{
  constexpr int dim = 2;
  double error = patch_test<dim>(affine_solution<dim>);
  EXPECT_LT(error, 1e-10);
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

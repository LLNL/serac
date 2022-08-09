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

using solid_mechanics::default_dynamic_options;
using solid_mechanics::default_static_options;

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

  auto body_force_function = [](auto /* x */, auto /* t */) { return tensor<double, dim>{}; };

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
  auto mesh =
      mesh::refineAndDistribute(buildHypercubeMesh<dim>(elements_per_dim), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  auto linear_solver_options        = solid_mechanics::default_linear_options;
  linear_solver_options.rel_tol     = 1e-7;
  linear_solver_options.abs_tol     = 1e-16;
  linear_solver_options.print_level = 0;
  const SolverOptions options{linear_solver_options, solid_mechanics::default_nonlinear_options};

  // Construct a functional-based solid mechanics solver
  SolidFunctional<p, dim> solid_solver(options, GeometricNonlinearities::On, FinalMeshOption::Reference,
                                       "solid_functional");

  double                      density       = 1.0;
  double                      shear_modulus = 1.0;
  double                      bulk_modulus  = 1.0;
  solid_mechanics::NeoHookean mat{density, shear_modulus, bulk_modulus};
  solid_solver.setMaterial(mat);

  // Set the initial displacement
  // solid_solver.setDisplacement([](const mfem::Vector&X, mfem::Vector& u) {});

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

  solid_solver.addBodyForce(body_force_function);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  double dt = 1.0;
  solid_solver.advanceTimestep(dt);

  // Output the sidre-based and paraview plot files
  solid_solver.outputState("paraview_output");

  // Compute norm of error
  mfem::VectorFunctionCoefficient exact_solution_coef(dim, exact_displacement_function);
  double                          uNorm = solid_solver.displacement().gridFunction().Norml2();
  std::cout << "||u|| = " << uNorm << std::endl;
  std::cout << "error = " << solid_solver.displacement().gridFunction().ComputeL2Error(exact_solution_coef)
            << std::endl;
  return solid_solver.displacement().gridFunction().ComputeL2Error(exact_solution_coef);
}

template <int dim>
double patch_test_linear(std::function<void(const mfem::Vector&, mfem::Vector&)> exact_displacement_function)
{
  constexpr int p = 1;

  // auto body_force_function = [](auto, auto, auto, auto){ return tensor<double, dim>{}; };

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
  auto mymesh = buildHypercubeMesh<dim>(elements_per_dim);
  mymesh.Save("garbage.txt");
  auto mesh =
      mesh::refineAndDistribute(buildHypercubeMesh<dim>(elements_per_dim), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  const DirectSolverOptions linear_solver_options = {.print_level = 1};
  // const NonlinearSolverOptions nonlinear_solver_options = {
  //    .rel_tol = 1.0e-10, .abs_tol = 1.0e-15, .max_iter = 1, .print_level = 1};
  // auto linear_solver_options = solid_mechanics::default_linear_options;
  // linear_solver_options.rel_tol=1e-11;
  // linear_solver_options.abs_tol=1e-16;
  auto nonlinear_solver_options = solid_mechanics::default_nonlinear_options;

  const SolverOptions static_solver_options = {linear_solver_options, nonlinear_solver_options};

  // Construct a functional-based solid mechanics solver
  SolidFunctional<p, dim> solid_solver(static_solver_options, GeometricNonlinearities::Off, FinalMeshOption::Reference,
                                       "solid_functional");

  double                           density       = 1.0;
  double                           shear_modulus = 1.0;
  double                           bulk_modulus  = 1.0;
  solid_mechanics::LinearIsotropic mat{density, shear_modulus, bulk_modulus};
  solid_solver.setMaterial(mat);

  // Set the initial displacement
  // solid_solver.setDisplacement([](const mfem::Vector&, mfem::Vector& u) { u = 0.0; });

  // Define a boundary attribute set
  int           x_equals_0 = (dim == 2) ? 4 : 0;
  int           y_equals_0 = (dim == 2) ? 1 : 0;
  std::set<int> essential_boundaries;
  if constexpr (dim == 2) {
    essential_boundaries = {x_equals_0, y_equals_0};
  } else {
    int z_equals_0       = 5;
    essential_boundaries = {x_equals_0, y_equals_0, z_equals_0};
  }

  // displacement boundary condition
  auto x_bc = [&](auto x) {
    mfem::Vector u(2);
    exact_displacement_function(x, u);
    return u[0];
  };
  solid_solver.setDisplacementBCs({x_equals_0}, x_bc, 0);

  auto y_bc = [&](auto x) {
    mfem::Vector u(2);
    exact_displacement_function(x, u);
    return u[1];
  };
  solid_solver.setDisplacementBCs({y_equals_0}, y_bc, 1);

  // traction
  solid_solver.setPiolaTraction([](auto x, auto, auto) {
    return tensor<double, 2>{{(x[0] == 1) ? 2.2857142857142856 : 0.0, 0.0}};
  });

  // solid_solver.addBodyForce(force);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  double dt = 1.0;
  solid_solver.advanceTimestep(dt);

  // Output the sidre-based and paraview plot files
  solid_solver.outputState("paraview_output");

  // Compute norm of error
  mfem::VectorFunctionCoefficient exact_solution_coef(dim, exact_displacement_function);
  double                          uNorm = solid_solver.displacement().gridFunction().Norml2();
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

  auto options              = default_static_options;
  options.nonlinear.rel_tol = 1.0e-8;
  options.nonlinear.abs_tol = 1.0e-16;

  // Construct a functional-based solid mechanics solver
  SolidFunctional<p, dim> solid_solver(options, GeometricNonlinearities::On, FinalMeshOption::Reference,
                                       "solid_functional");

  solid_mechanics::NeoHookean mat{1.0, 1.0, 1.0};
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

  solid_mechanics::ConstantBodyForce<dim> force{constant_force};
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

void functional_solid_test_static_J2()
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int p                   = 2;
  constexpr int dim                 = 3;
  int           serial_refinement   = 0;
  int           parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_functional_static_solve_J2");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  auto options           = default_static_options;
  auto linear_options    = solid_mechanics::default_linear_options;
  linear_options.abs_tol = 1.0e-16;  // prevent early-exit in linear solve
  options.linear         = linear_options;

  // Construct a functional-based solid mechanics solver
  SolidFunctional<p, dim> solid_solver(options, GeometricNonlinearities::Off, FinalMeshOption::Reference,
                                       "solid_functional");

  solid_mechanics::J2 mat{
      10000,  // Young's modulus
      0.25,   // Poisson's ratio
      50.0,   // isotropic hardening constant
      5.0,    // kinematic hardening constant
      50.0,   // yield stress
      1.0     // mass density
  };

  solid_mechanics::J2::State initial_state{};

  auto state = solid_solver.createQuadratureDataBuffer(initial_state);

  solid_solver.setMaterial(mat, state);

  // prescribe zero displacement at the supported end of the beam,
  std::set<int> support           = {1};
  auto          zero_displacement = [](const mfem::Vector&, mfem::Vector& u) -> void { u = 0.0; };
  solid_solver.setDisplacementBCs(support, zero_displacement);

  // apply a displacement along z to the the tip of the beam
  auto translated_in_z = [](const mfem::Vector&, double t, mfem::Vector& u) -> void {
    u    = 0.0;
    u[2] = t * (t - 1);
  };
  std::set<int> tip = {2};
  solid_solver.setDisplacementBCs(tip, translated_in_z);

  solid_solver.setDisplacement(zero_displacement);

  // Finalize the data structures
  solid_solver.completeSetup();

  solid_solver.outputState("paraview");

  // Perform the quasi-static solve
  int    num_steps = 10;
  double tmax      = 1.0;
  double dt        = tmax / num_steps;
  for (int i = 0; i < num_steps; i++) {
    solid_solver.advanceTimestep(dt);
    solid_solver.outputState("paraview");
  }

  // this a qualitative test that just verifies
  // that plasticity models can have permanent
  // deformation after unloading
  // EXPECT_LT(norm(solid_solver.nodalForces()), 1.0e-5);
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

  // Construct a functional-based solid mechanics solver
  SolidFunctional<p, dim> solid_solver(default_dynamic_options, GeometricNonlinearities::Off,
                                       FinalMeshOption::Reference, "solid_functional_dynamic");

  solid_mechanics::LinearIsotropic mat{1.0, 1.0, 1.0};
  solid_solver.setMaterial(mat);

  // Define the function for the initial displacement and boundary condition
  auto bc = [](const mfem::Vector&, mfem::Vector& bc_vec) -> void { bc_vec = 0.0; };

  // Define a boundary attribute set and specify initial / boundary conditions
  std::set<int> ess_bdr = {1};
  solid_solver.setDisplacementBCs(ess_bdr, bc);
  solid_solver.setDisplacement(bc);

  tensor<double, dim> constant_force{0.0, 0.5};

  solid_mechanics::ConstantBodyForce<dim> force{constant_force};
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

  // Construct a functional-based solid mechanics solver
  SolidFunctional<p, dim> solid_solver(default_static_options, GeometricNonlinearities::Off, FinalMeshOption::Reference,
                                       "solid_functional");

  solid_mechanics::LinearIsotropic mat{1.0, 1.0, 1.0};
  solid_solver.setMaterial(mat);

  // Define the function for the initial displacement and boundary condition
  auto bc = [](const mfem::Vector&, mfem::Vector& bc_vec) -> void { bc_vec = 0.0; };

  // Define a boundary attribute set and specify initial / boundary conditions
  std::set<int> ess_bdr = {1};
  solid_solver.setDisplacementBCs(ess_bdr, bc);
  solid_solver.setDisplacement(bc);

  if (test_mode == TestType::Pressure) {
    solid_solver.setPiolaTraction([](const tensor<double, dim>& x, const tensor<double, dim>& n, const double) {
      if (x[0] > 7.5) {
        return 1.0e-2 * n;
      }
      return 0.0 * n;
    });
  } else if (test_mode == TestType::Traction) {
    solid_solver.setPiolaTraction([](const tensor<double, dim>& x, const tensor<double, dim>& /*n*/, const double) {
      tensor<double, dim> traction;
      for (int i = 0; i < dim; ++i) {
        traction[i] = (x[0] > 7.9) ? 1.0e-4 : 0.0;
      }
      return traction;
    });
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

  // Construct and initialized the user-defined moduli to be used as a differentiable parameter in
  // the solid mechanics physics module.
  FiniteElementState user_defined_shear_modulus(
      StateManager::newState(FiniteElementState::Options{.order = 1, .name = "parameterized_shear"}));

  user_defined_shear_modulus = 1.0;

  FiniteElementState user_defined_bulk_modulus(
      StateManager::newState(FiniteElementState::Options{.order = 1, .name = "parameterized_bulk"}));

  user_defined_bulk_modulus = 1.0;

  // Construct a functional-based solid mechanics solver
  SolidFunctional<p, dim, Parameters<H1<1>, H1<1>>> solid_solver(
      default_static_options, GeometricNonlinearities::On, FinalMeshOption::Reference, "solid_functional",
      {user_defined_bulk_modulus, user_defined_shear_modulus});

  solid_mechanics::ParameterizedNeoHookeanSolid<dim> mat{1.0, 0.0, 0.0};
  solid_solver.setMaterial(mat);

  // Define the function for the initial displacement and boundary condition
  auto bc = [](const mfem::Vector&, mfem::Vector& bc_vec) -> void { bc_vec = 0.0; };

  // Define a boundary attribute set and specify initial / boundary conditions
  std::set<int> ess_bdr = {1};
  solid_solver.setDisplacementBCs(ess_bdr, bc);
  solid_solver.setDisplacement(bc);

  tensor<double, dim> constant_force;

  constant_force[0] = 0.0;
  constant_force[1] = 5.0e-4;

  if (dim == 3) {
    constant_force[2] = 0.0;
  }

  solid_mechanics::ConstantBodyForce<dim> force{constant_force};
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

TEST(SolidFunctional, 2DLinearStatic) { functional_solid_test_static<1, 2>(1.5110743593501033); }
TEST(SolidFunctional, 2DQuadStatic) { functional_solid_test_static<2, 2>(2.1864070695817928); }
TEST(SolidFunctional, 2DQuadParameterizedStatic) { functional_parameterized_solid_test<2, 2>(2.1864070695817928); }

TEST(SolidFunctional, 3DLinearStatic) { functional_solid_test_static<1, 3>(1.3708718127987922); }
TEST(SolidFunctional, 3DQuadStatic) { functional_solid_test_static<2, 3>(1.9497254957351946); }

TEST(SolidFunctional, 3DQuadStaticJ2) { functional_solid_test_static_J2(); }

TEST(SolidFunctional, 2DLinearDynamic) { functional_solid_test_dynamic<1, 2>(1.52116682); }
TEST(SolidFunctional, 2DQuadDynamic) { functional_solid_test_dynamic<2, 2>(1.52777214); }

TEST(SolidFunctional, 3DLinearDynamic) { functional_solid_test_dynamic<1, 3>(1.520679017); }
TEST(SolidFunctional, 3DQuadDynamic) { functional_solid_test_dynamic<2, 3>(1.527009514); }

TEST(SolidFunctional, 2DLinearPressure) { functional_solid_test_boundary<1, 2>(0.065326222, TestType::Pressure); }
TEST(SolidFunctional, 2DLinearTraction)
{
  functional_solid_test_boundary<1, 2>(0.12659525750241674, TestType::Traction);
}

template <int dim>
void affine_solution(const mfem::Vector& X, mfem::Vector u)
{
  mfem::DenseMatrix A(dim);
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

  mfem::Vector b(dim);
  b(0) = 0.765645367640828;
  b(1) = 0.992487355850465;
  if constexpr (dim == 3) {
    b(2) = 0.162199373722092;
  }

  A.Mult(X, u);
  u += b;
}

TEST(SolidFunctional, Patch2D)
{
  constexpr int dim   = 2;
  double        error = patch_test<dim>(affine_solution<dim>);
  EXPECT_LT(error, 1e-10);
}

TEST(SolidFunctional, Patch2DLinear)
{
  auto linear_solution = [](const mfem::Vector& X, mfem::Vector& u) {
    u(0) = X[0];
    u(1) = -1.0 / 7.0 * X[1];
  };
  constexpr int dim   = 2;
  double        error = patch_test_linear<dim>(linear_solution);
  std::cout << "errror " << error << std::endl;
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

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
  SolidFunctional<p, dim> solid_solver(default_static, GeometricNonlinearities::On, FinalMeshOption::Reference,
                                       "solid_functional");

  solid_util::NeoHookeanSolid<dim> mat{1.0, 1.0, 1.0};
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

void functional_solid_test_static_J2(double expected_disp_norm)
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int p = 2;
  constexpr int dim = 3;
  int serial_refinement   = 0;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_functional_static_solve_J2");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";

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
  SolidFunctional<p, dim> solid_solver(default_static, GeometricNonlinearities::On, FinalMeshOption::Reference,
                                       "solid_functional");

  solid_util::J2 mat{
    100,   // Young's modulus
    0.25,  // Poisson's ratio
    1.0,   // isotropic hardening constant
    2.3,   // kinematic hardening constant
    300.0, // yield stress
    1.0    // mass density
  };

  solid_util::J2::State initial_state{};

  auto state = solid_solver.createQuadratureDataBuffer(initial_state);

  solid_solver.setMaterial(mat, state);

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

// --------------------------------------------------------

void functional_solid_test_lce_material(double expected_disp_norm)
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int p = 2;
  constexpr int dim = 3;
  int serial_refinement   = 0;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "lce_solid_functional_static_solve_J2");

  // Construct the appropriate dimension mesh and give it to the data store
  // std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";
  // std::string filename = SERAC_REPO_DIR "/data/meshes/LCE_tensileTestSpecimen.g";
  std::string filename = SERAC_REPO_DIR "/data/meshes/LCE_tensileTestSpecimen_nonDim.g";
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  // auto mesh = serac::mesh::refineAndDistribute(serac::buildCuboidMesh(10, 10, 3, 0.008, 0.008, 0.00016));
  // serac::StateManager::setMesh(std::move(mesh));

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
  SolidFunctional<p, dim> solid_solver(default_static, GeometricNonlinearities::On, FinalMeshOption::Reference,
                                       "lce_solid_functional");

  solid_util::J2 mat{
    100,   // Young's modulus
    0.25,  // Poisson's ratio
    1.0,   // isotropic hardening constant
    2.3,   // kinematic hardening constant
    300.0, // yield stress
    1.0    // mass density
  };

// std::cout<<"... testing"<<std::endl;
  solid_util::J2::State initial_state{};

  auto state = solid_solver.createQuadratureDataBuffer(initial_state);

  solid_solver.setMaterial(mat, state);

  // Define the function for the initial displacement and boundary condition
  auto bc = [](const mfem::Vector&, mfem::Vector& bc_vec) -> void { bc_vec = 0.0; };

  // set the boundary conditions to be fixed on the coordinate planes
  auto zeroFunc = [](const mfem::Vector /*x*/){ return 0.0;};

  solid_solver.setDisplacementBCs({1}, zeroFunc, 1); // bottom face y-dir disp = 0
  solid_solver.setDisplacementBCs({2}, zeroFunc, 0); // left face x-dir disp = 0
  solid_solver.setDisplacementBCs({3}, zeroFunc, 2); // back face z-dir disp = 0

  solid_solver.setDisplacement(bc);

  bool includeBodyForce(false);

  if(includeBodyForce)
  {
    tensor<double, dim> constant_force;

    constant_force[0] = 0.0;
    constant_force[1] = -8.0e-2;

    if (dim == 3) {
      constant_force[2] = 0.0;
    }

    solid_util::ConstantBodyForce<dim> force{constant_force};
    solid_solver.addBodyForce(force); 
  }

  solid_solver.setPiolaTraction([](auto x, auto /*n*/, auto t){
    return tensor<double, 3>{0, 5.0e-3 * (x[1] > 0.0079), 0*t};
  });

  // Finalize the data structures
  solid_solver.completeSetup();
  
  // Perform the quasi-static solve
  double dt = 1.0;
  solid_solver.advanceTimestep(dt);

  // Output the sidre-based plot files
  solid_solver.outputState();

  solid_solver.outputState("lce_paraview_output");

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

  solid_util::LinearIsotropicSolid<dim> mat{1.0, 1.0, 1.0};
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

  solid_util::LinearIsotropicSolid<dim> mat{1.0, 1.0, 1.0};
  solid_solver.setMaterial(mat);

  // Define the function for the initial displacement and boundary condition
  auto bc = [](const mfem::Vector&, mfem::Vector& bc_vec) -> void { bc_vec = 0.0; };

  // Set the initial displacement and boundary condition
  solid_solver.setDisplacementBCs(ess_bdr, bc);
  solid_solver.setDisplacement(bc);

  if (test_mode == TestType::Pressure) {
    solid_solver.setPiolaTraction([](const tensor<double, dim>& x, const tensor<double, dim> & n, const double) {
      if (x[0] > 7.5) {
        return 1.0e-2 * n;
      }
      return 0.0 * n;
    });
  } else if (test_mode == TestType::Traction) {
    solid_solver.setPiolaTraction([](const tensor<double, dim>& x, const tensor<double, dim> & /*n*/, const double) {
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
  SolidFunctional<p, dim, Parameters<H1<1>, H1<1>> > solid_solver(default_static, GeometricNonlinearities::On,
                                                     FinalMeshOption::Reference, "solid_functional",
                                                     {user_defined_bulk_modulus, user_defined_shear_modulus});

  solid_util::ParameterizedNeoHookeanSolid<dim> mat{1.0, 0.0, 0.0};
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

//TEST(SolidFunctional, 2DLinearStatic) { functional_solid_test_static<1, 2>(1.511052595); }
//TEST(SolidFunctional, 2DQuadStatic) { functional_solid_test_static<2, 2>(2.18604855); }
//TEST(SolidFunctional, 2DQuadParameterizedStatic) { functional_parameterized_solid_test<2, 2>(2.18604855); }
//
//TEST(SolidFunctional, 3DLinearStatic) { functional_solid_test_static<1, 3>(1.37084852); }
//TEST(SolidFunctional, 3DQuadStatic) { functional_solid_test_static<2, 3>(1.949532747); }
//
TEST(SolidFunctional, 3DQuadStaticJ2) { functional_solid_test_lce_material(0.0); }
//
// TEST(SolidFunctional, 3DQuadStaticJ2) { functional_solid_test_static_J2(0.0); }
//
//TEST(SolidFunctional, 2DLinearDynamic) { functional_solid_test_dynamic<1, 2>(1.52116682); }
//TEST(SolidFunctional, 2DQuadDynamic) { functional_solid_test_dynamic<2, 2>(1.52777214); }
//
//TEST(SolidFunctional, 3DLinearDynamic) { functional_solid_test_dynamic<1, 3>(1.520679017); }
//TEST(SolidFunctional, 3DQuadDynamic) { functional_solid_test_dynamic<2, 3>(1.527009514); }
//
//TEST(SolidFunctional, 2DLinearPressure) { functional_solid_test_boundary<1, 2>(0.065326222, TestType::Pressure); }
//TEST(SolidFunctional, 2DLinearTraction) { functional_solid_test_boundary<1, 2>(0.126593590, TestType::Traction); }

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

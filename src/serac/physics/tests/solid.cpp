// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_mechanics.hpp"

#include <functional>
#include <fstream>
#include <set>
#include <string>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/physics/materials/parameterized_solid_material.hpp"
#include "serac/serac_config.hpp"

namespace serac {

void functional_solid_test_static_J2()
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int p                   = 2;
  constexpr int dim                 = 3;
  int           serial_refinement   = 0;
  int           parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_mechanics_J2_test");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  auto linear_options         = solid_mechanics::direct_linear_options;
  linear_options.absolute_tol = 1.0e-16;  // prevent early-exit in linear solve

  // Construct a functional-based solid mechanics solver
  SolidMechanics<p, dim> solid_solver(solid_mechanics::default_nonlinear_options, linear_options,
                                      solid_mechanics::default_quasistatic_options, GeometricNonlinearities::Off,
                                      "solid_mechanics");

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
  // EXPECT_LT(norm(solid_solver.reactions()), 1.0e-5);
}

// The purpose of this test is to check that the spatial function-defined essential boundary conditions are
// working appropriately. It takes a 4 hex cube mesh and pins it in one corner. The z-direction displacement
// is set to zero on the bottom face and a constant negative value on the top face.
void functional_solid_spatial_essential_bc()
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int p                   = 1;
  constexpr int dim                 = 3;
  int           serial_refinement   = 1;
  int           parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_mechanics_spatial_essential");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/onehex.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  // Construct a functional-based solid mechanics solver
  SolidMechanics<p, dim> solid_solver(
      solid_mechanics::default_nonlinear_options, solid_mechanics::default_linear_options,
      solid_mechanics::default_quasistatic_options, GeometricNonlinearities::Off, "solid_mechanics");

  solid_mechanics::LinearIsotropic mat{1.0, 1.0, 1.0};
  solid_solver.setMaterial(mat);

  // Set up
  auto zero_vector   = [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; };
  auto zero_scalar   = [](const mfem::Vector&) { return 0.0; };
  auto scalar_offset = [](const mfem::Vector&) { return -0.1; };

  auto is_on_bottom = [](const mfem::Vector& x) {
    if (x(2) < 0.01) {
      return true;
    }
    return false;
  };

  auto is_on_bottom_corner = [](const mfem::Vector& x) {
    if (x(0) < 0.01 && x(1) < 0.01 && x(2) < 0.01) {
      return true;
    }
    return false;
  };

  auto is_on_top = [](const mfem::Vector& x) {
    if (x(2) > 0.95) {
      return true;
    }
    return false;
  };

  solid_solver.setDisplacementBCs(is_on_bottom_corner, zero_vector);
  solid_solver.setDisplacementBCs(is_on_bottom, zero_scalar, 2);
  solid_solver.setDisplacementBCs(is_on_top, scalar_offset, 2);

  // Set a zero initial guess
  solid_solver.setDisplacement(zero_vector);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  double dt = 1.0;
  solid_solver.advanceTimestep(dt);
  solid_solver.outputState();

  auto [size, rank] = serac::getMPIInfo();

  // This exact solution is only correct when two MPI ranks are used
  // It is based on a poisson ratio of 0.125 with a prescribed z strain of 10%
  if (size == 2) {
    mfem::Vector rank_0_displacement(54);
    rank_0_displacement(0)  = 0.0;
    rank_0_displacement(1)  = 0.0125;
    rank_0_displacement(2)  = 0.00444899764940326;
    rank_0_displacement(3)  = 0.01694899764940309;
    rank_0_displacement(4)  = 0.006249999999997622;
    rank_0_displacement(5)  = 0.014724498824700673;
    rank_0_displacement(6)  = 0.010698997649401977;
    rank_0_displacement(7)  = 0.0022244988247011586;
    rank_0_displacement(8)  = 0.0;
    rank_0_displacement(9)  = 0.0125;
    rank_0_displacement(10) = 0.016948997649402067;
    rank_0_displacement(11) = 0.004448997649403503;
    rank_0_displacement(12) = 0.006249999999998786;
    rank_0_displacement(13) = 0.014724498824701405;
    rank_0_displacement(14) = 0.010698997649404096;
    rank_0_displacement(15) = 0.002224498824700543;
    rank_0_displacement(16) = 0.008474498824701965;
    rank_0_displacement(17) = 0.008474498824699537;
    rank_0_displacement(18) = 0.0;
    rank_0_displacement(19) = -0.004448997649405797;
    rank_0_displacement(20) = 0.0125;
    rank_0_displacement(21) = 0.00805100235059582;
    rank_0_displacement(22) = -0.0022244988247037854;
    rank_0_displacement(23) = 0.0018010023505955542;
    rank_0_displacement(24) = 0.010275501175296586;
    rank_0_displacement(25) = 0.00624999999999837;
    rank_0_displacement(26) = 0.0;
    rank_0_displacement(27) = -0.0044489976494050025;
    rank_0_displacement(28) = 0.008051002350592475;
    rank_0_displacement(29) = 0.0125;
    rank_0_displacement(30) = -0.002224498824703152;
    rank_0_displacement(31) = 0.0018010023505938145;
    rank_0_displacement(32) = 0.010275501175297441;
    rank_0_displacement(33) = 0.006250000000000508;
    rank_0_displacement(34) = 0.004025501175297109;
    rank_0_displacement(35) = 0.004025501175296371;
    rank_0_displacement(36) = -0.1;
    rank_0_displacement(37) = -0.1;
    rank_0_displacement(38) = -0.1;
    rank_0_displacement(39) = -0.1;
    rank_0_displacement(40) = -0.1;
    rank_0_displacement(41) = -0.1;
    rank_0_displacement(42) = -0.1;
    rank_0_displacement(43) = -0.1;
    rank_0_displacement(44) = -0.05;
    rank_0_displacement(45) = -0.05;
    rank_0_displacement(46) = -0.05;
    rank_0_displacement(47) = -0.05;
    rank_0_displacement(48) = -0.05;
    rank_0_displacement(49) = -0.05;
    rank_0_displacement(50) = -0.05;
    rank_0_displacement(51) = -0.05;
    rank_0_displacement(52) = -0.1;
    rank_0_displacement(53) = -0.05;

    mfem::Vector rank_1_displacement(27);
    rank_1_displacement(0)  = 0.0;
    rank_1_displacement(1)  = 0.0125;
    rank_1_displacement(2)  = 0.004448997649402821;
    rank_1_displacement(3)  = 0.016948997649407428;
    rank_1_displacement(4)  = 0.006249999999997266;
    rank_1_displacement(5)  = 0.014724498824698999;
    rank_1_displacement(6)  = 0.010698997649403928;
    rank_1_displacement(7)  = 0.0022244988247001056;
    rank_1_displacement(8)  = 0.008474498824701611;
    rank_1_displacement(9)  = 0.0;
    rank_1_displacement(10) = -0.004448997649407356;
    rank_1_displacement(11) = 0.0125;
    rank_1_displacement(12) = 0.008051002350593363;
    rank_1_displacement(13) = -0.0022244988247036765;
    rank_1_displacement(14) = 0.0018010023505931603;
    rank_1_displacement(15) = 0.01027550117529734;
    rank_1_displacement(16) = 0.006249999999998952;
    rank_1_displacement(17) = 0.004025501175297568;
    rank_1_displacement(18) = 0.0;
    rank_1_displacement(19) = 0.0;
    rank_1_displacement(20) = 0.0;
    rank_1_displacement(21) = 0.0;
    rank_1_displacement(22) = 0.0;
    rank_1_displacement(23) = 0.0;
    rank_1_displacement(24) = 0.0;
    rank_1_displacement(25) = 0.0;
    rank_1_displacement(26) = 0.0;

    if (rank == 0) {
      for (int i = 0; i < solid_solver.displacement().Size(); ++i) {
        EXPECT_NEAR(rank_0_displacement(i), solid_solver.displacement()(i), 1.0e-8);
      }
    }

    if (rank == 1) {
      for (int i = 0; i < solid_solver.displacement().Size(); ++i) {
        EXPECT_NEAR(rank_1_displacement(i), solid_solver.displacement()(i), 1.0e-8);
      }
    }
  }
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

  // _solver_params_start
  serac::LinearSolverOptions linear_options{.linear_solver  = LinearSolver::GMRES,
                                            .preconditioner = Preconditioner::HypreAMG,
                                            .relative_tol   = 1.0e-6,
                                            .absolute_tol   = 1.0e-14,
                                            .max_iterations = 500,
                                            .print_level    = 1};

#ifdef MFEM_USE_SUNDIALS
  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver  = NonlinearSolver::KINFullStep,
                                                  .relative_tol   = 1.0e-12,
                                                  .absolute_tol   = 1.0e-12,
                                                  .max_iterations = 5000,
                                                  .print_level    = 1};
#else
  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver  = NonlinearSolver::Newton,
                                                  .relative_tol   = 1.0e-12,
                                                  .absolute_tol   = 1.0e-12,
                                                  .max_iterations = 5000,
                                                  .print_level    = 1};
#endif

  SolidMechanics<p, dim> solid_solver(nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options,
                                      GeometricNonlinearities::Off, "solid_mechanics");
  // _solver_params_end

  solid_mechanics::LinearIsotropic mat{1.0, 1.0, 1.0};
  solid_solver.setMaterial(mat);

  // Define the function for the initial displacement and boundary condition
  auto bc = [](const mfem::Vector&, mfem::Vector& bc_vec) -> void { bc_vec = 0.0; };

  // Define a boundary attribute set and specify initial / boundary conditions
  std::set<int> ess_bdr = {1};
  solid_solver.setDisplacementBCs(ess_bdr, bc);
  solid_solver.setDisplacement(bc);

  if (test_mode == TestType::Pressure) {
    solid_solver.setPiolaTraction([](const auto& x, const tensor<double, dim>& n, const double) {
      if (x[0] > 7.5) {
        return 5.0e-3 * n;
      }
      return 0.0 * n;
    });
  } else if (test_mode == TestType::Traction) {
    solid_solver.setPiolaTraction([](const auto& x, const tensor<double, dim>& /*n*/, const double) {
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

template <typename lambda>
struct ParameterizedBodyForce {
  template <int dim, typename T1, typename T2>
  auto operator()(const tensor<T1, dim> x, double /*t*/, T2 density) const
  {
    return get<0>(density) * acceleration(x);
  }
  lambda acceleration;
};

template <typename T>
ParameterizedBodyForce(T) -> ParameterizedBodyForce<T>;

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

  auto  mesh  = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  auto* pmesh = serac::StateManager::setMesh(std::move(mesh));

  // Construct and initialized the user-defined moduli to be used as a differentiable parameter in
  // the solid mechanics physics module.
  FiniteElementState user_defined_shear_modulus(
      StateManager::newState(FiniteElementState::Options{.order = 1, .name = "parameterized_shear"}));

  user_defined_shear_modulus = 1.0;

  FiniteElementState user_defined_bulk_modulus(
      StateManager::newState(FiniteElementState::Options{.order = 1, .name = "parameterized_bulk"}));

  user_defined_bulk_modulus = 1.0;

  // _custom_solver_start
  auto nonlinear_solver = std::make_unique<mfem::NewtonSolver>(pmesh->GetComm());
  nonlinear_solver->SetPrintLevel(1);
  nonlinear_solver->SetMaxIter(30);
  nonlinear_solver->SetAbsTol(1.0e-12);
  nonlinear_solver->SetRelTol(1.0e-10);

  auto linear_solver = std::make_unique<mfem::HypreGMRES>(pmesh->GetComm());
  linear_solver->SetPrintLevel(1);
  linear_solver->SetMaxIter(500);
  linear_solver->SetTol(1.0e-6);

  auto preconditioner = std::make_unique<mfem::HypreBoomerAMG>();
  linear_solver->SetPreconditioner(*preconditioner);

  auto equation_solver = std::make_unique<EquationSolver>(std::move(nonlinear_solver), std::move(linear_solver),
                                                          std::move(preconditioner));

  SolidMechanics<p, dim, Parameters<H1<1>, H1<1>>> solid_solver(std::move(equation_solver),
                                                                solid_mechanics::default_quasistatic_options,
                                                                GeometricNonlinearities::On, "parameterized_solid");
  // _custom_solver_end

  solid_solver.setParameter(0, user_defined_bulk_modulus);
  solid_solver.setParameter(1, user_defined_shear_modulus);

  solid_mechanics::ParameterizedNeoHookeanSolid<dim> mat{1.0, 0.0, 0.0};
  solid_solver.setMaterial(DependsOn<0, 1>{}, mat);

  // Define the function for the initial displacement and boundary condition
  auto bc = [](const mfem::Vector&, mfem::Vector& bc_vec) -> void { bc_vec = 0.0; };

  // Define a boundary attribute set and specify initial / boundary conditions
  std::set<int> ess_bdr = {1};

  // Generate a true dof set from the boundary attribute
  mfem::Array<int> bdr_attr_marker(pmesh->bdr_attributes.Max());
  bdr_attr_marker    = 0;
  bdr_attr_marker[0] = 1;
  mfem::Array<int> true_dofs;
  solid_solver.displacement().space().GetEssentialTrueDofs(bdr_attr_marker, true_dofs);

  solid_solver.setDisplacementBCsByDofList(true_dofs, bc);
  solid_solver.setDisplacement(bc);

  tensor<double, dim> constant_force;

  constant_force[0] = 0.0;
  constant_force[1] = 5.0e-4;

  if (dim == 3) {
    constant_force[2] = 0.0;
  }

  solid_mechanics::ConstantBodyForce<dim> force{constant_force};
  solid_solver.addBodyForce(force);

  // add some nonexistent body forces / tractions to check that
  // these parameterized versions compile and run without error
  solid_solver.addBodyForce(DependsOn<0>{}, [](const auto& x, double /*t*/, auto /* bulk */) { return x * 0.0; });
  solid_solver.addBodyForce(DependsOn<1>{}, ParameterizedBodyForce{[](const auto& x) { return 0.0 * x; }});
  solid_solver.setPiolaTraction(DependsOn<1>{}, [](const auto& x, auto...) { return 0 * x; });

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  double dt = 1.0;
  solid_solver.advanceTimestep(dt);

  // the calculations peformed in these lines of code
  // are not used, but running them as part of this test
  // checks the index-translation part of the derivative
  // kernels is working
  solid_solver.computeSensitivity(0);
  solid_solver.computeSensitivity(1);

  // Output the sidre-based plot files
  solid_solver.outputState();

  // Check the final displacement norm
  EXPECT_NEAR(expected_disp_norm, norm(solid_solver.displacement()), 1.0e-6);
}

TEST(SolidMechanics, 2DQuadParameterizedStatic) { functional_parameterized_solid_test<2, 2>(2.1906312704664623); }

TEST(SolidMechanics, 3DQuadStaticJ2) { functional_solid_test_static_J2(); }

TEST(SolidMechanics, SpatialBoundaryCondition) { functional_solid_spatial_essential_bc(); }

TEST(SolidMechanics, 2DLinearPressure)
{
  functional_solid_test_boundary<1, 2>(0.028525698834671667, TestType::Pressure);
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

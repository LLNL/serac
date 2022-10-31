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
#include "serac/physics/materials/solid_material.hpp"
#include "serac/physics/materials/parameterized_solid_material.hpp"

#include "serac/physics/materials/liquid_crystal_elastomer_material.hpp"

namespace serac {
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

//
TEST(SolidFunctional, 3DQuadStaticJ2) { functional_solid_test_lce_material(0.0); }
//
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

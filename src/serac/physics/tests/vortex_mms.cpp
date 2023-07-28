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
#include <cmath>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/serac_config.hpp"
#include "serac/numerics/functional/tensor.hpp"

namespace serac {

double compute_patch_test_error(int refinements) {
  constexpr int p   = 1;
  constexpr int dim = 3;

  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "vortex_mms_data");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/toroid-hex.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), refinements, 0);
  serac::StateManager::setMesh(std::move(mesh));

    serac::LinearSolverOptions linear_options{.linear_solver  = LinearSolver::GMRES,
                                            .preconditioner = Preconditioner::HypreAMG,
                                            .relative_tol   = 1.0e-6,
                                            .absolute_tol   = 1.0e-14,
                                            .max_iterations = 500,
                                            .print_level    = 1}; 


  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver  = NonlinearSolver::Newton,
                                                  .relative_tol   = 1.0e-9,
                                                  .absolute_tol   = 1.0e-12,
                                                  .max_iterations = 20,
                                                  .print_level    = 1};


    SolidMechanics<p, dim> solid_solver(nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options,
                                      GeometricNonlinearities::On, "solid_mechanics"); 
  //changed from direct to default for linear_options
 /* SolidMechanics<p, dim> solid_solver(nonlinear_options, serac::solid_mechanics::default_linear_options, solid_mechanics::default_quasistatic_options,
                                      GeometricNonlinearities::On, "solid_mechanics"); */
  double E = 1e3;
  double nu = 0.3;
  double                             K = E/(3*(1-2*nu));
  double                             G = E/(2*(1+nu));
  solid_mechanics::NeoHookean mat{1.0, K, G};
  solid_solver.setMaterial(mat);

  // from parameterized_thermomechanics_example.cpp
  // set up essential boundary conditions
  std::set<int> xy_equals_0 = {1};

  auto zero_vector = [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; };
  solid_solver.setDisplacementBCs(xy_equals_0, zero_vector);
;


  //body force
  auto body_force = [E,nu,G](const auto& x, const double) {
    using std::cos;
    using std::sin;
    using std::log;
    using std::pow;
    using std::atan;

    double t = 0.5;

    auto force=x*0.0;

    auto lambda=E*nu/(1+nu)/(1-2*nu);
    auto mu=G;
    auto A=3.1415*0.5;
    auto H=8;
    auto rho=1000;
    auto B = A*0.5*(1 - cos(2*3.1415*t));
    auto alpha = B*x[1]/H;
    auto R = sqrt(pow(x[0],2)+pow(x[1],2));
    auto phi = atan(x,y);
    double pi = 3.1415;
    auto p1 = 4096*R*pow(15-47*R+48*pow(R,2)-16*pow(R,3),2)*mu*pow(sin(2*pi*t),4)/rho;
    auto p2 = pow(pi,2)*R*pow(15-32*R+16*pow(R,2),4)*pow(sin(2*pi*t),2);
    auto p3 = -16*(-45+188*R-240*pow(R,2)+96*pow(R,3));
    auto p4 = -45+188*R-240*pow(R,2)+96*pow(R,3);
    auto p5 = pow(15-32*R+16*pow(R,2),2);
    auto br = p1-p2;
    auto b_theta = (2*mu*p3+2*cos(2*pi*t)*(16*mu*p4+pow(pi,2)*R*rho*p5))/rho;
    auto alpha = 0.5*A*(1-cos(2*pi*t))*(1-32*pow(R-1,2)+256**pow(R-1,4));
    auto theta = phi + alpha;
    auto bx = br*cos(theta) - b_theta*sin(theta);
    auto by = br*sin(theta) + b_theta*cos(theta);
    auto bz =0;
      
    
   force(0)=bx;
   force(1)=by;
   force (2)=bz;
    

   return force*t;
  };
  solid_solver.addBodyForce(body_force);  

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  double dt = 1;
  solid_solver.advanceTimestep(dt);

  // Output the sidre-based plot files
  solid_solver.outputState("visit_output");


  auto exact_disp = [](const mfem::Vector& X, mfem::Vector& u) {
    // u = x - X, where x = 2*X + 0*Y + 0*Z
    u[0] = X[0];
    u[1] = 0;
    u[2] = 0;
  };

  // Compute norm of error
  mfem::VectorFunctionCoefficient exact_solution_coef(dim, exact_disp);
  return computeL2Error(solid_solver.displacement(), exact_solution_coef);

}

TEST(Manufactured, Patch2D) {
  // call compute_patch_test_error
  double error = compute_patch_test_error(2);
  // check error
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

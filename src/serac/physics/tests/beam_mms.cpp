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
#include "serac/serac_config.hpp"
#include "serac/numerics/functional/tensor.hpp"

namespace serac {

double compute_patch_test_error(int refinements) {
  
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int p   = 1;
  constexpr int dim = 2;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "beam_mms_data");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/more_meshes/beam_tall.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), refinements, 0);
  serac::StateManager::setMesh(std::move(mesh));

    /* serac::LinearSolverOptions linear_options{.linear_solver  = LinearSolver::GMRES,
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
                                      GeometricNonlinearities::On, "solid_mechanics"); */

  //changed from direct to default for linear_options
  SolidMechanics<p, dim> solid_solver(solid_mechanics::default_nonlinear_options,
    solid_mechanics::default_linear_options, solid_mechanics::default_quasistatic_options,
    GeometricNonlinearities::On, "solid_mechanics"); 

  double E = 1e3;
  double nu = 0.3;
  double                             K = E/(3*(1-2*nu));
  double                             G = E/(2*(1+nu));
  solid_mechanics::NeoHookean mat{1.0, K, G};
  solid_solver.setMaterial(mat);

  // from parameterized_thermomechanics_example.cpp
  // set up essential boundary conditions
  std::set<int> xy_equals_0 = {1};

  //auto zero_scalar = [](const mfem::Vector&) -> double { return 0.0; };
  auto zero_vector = [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; };
  solid_solver.setDisplacementBCs(xy_equals_0, zero_vector);
  //solid_solver.setDisplacementBCs(xy_equals_0, zero_scalar, 1);
;


  //body force
  auto body_force = [E,nu,G](const auto& x, const double /*t*/) {
    /*using std::cos;
    using std::sin;
    using std::log;
    using std::pow;

    //double t = 0.5;

    auto force=x*0.0;

    auto lambda=E*nu/(1+nu)/(1-2*nu);
    auto mu=G;
    auto A=3.1415*0.5;
    auto H=8;
    auto rho=1000;
    auto B = A*0.5*(1 - cos(2*3.1415*t));
    auto alpha = B*x[1]/H;
    auto p1 = 128*pow(H,3)-8*pow(A,2)*H*pow(x[1],2)-5*pow(A,3)*x[0]*x[1]+4*(16*pow(H,3)+pow(A,2)*pow(x[1],2)+ 
	        pow(A,3)*x[0]*pow(x[1],2))*cos(2*3.1415*t);
    auto p2 = 4*pow(A,2)*(2*H + A*x[0])*x[1]*cos(4*3.1415*t)-4*pow(A,3)*x[0]*pow(x[1],2)*cos(6*3.1415*t)+pow(A,3)*x[0]*pow(x[1],
	        2)*cos(8*3.1415*t);
    //std::cout<< "p2="<< p2 <<std::endl;
    auto p3 = -128*pow(H,3)*cos((A*x[1]*pow(sin(3.1415*t),2))/H)-328*pow(H,3)*cos(2*3.1415*t-((A*x[1]*pow(sin(3.1415*t),2))/H));
    auto p4 = -32*pow(H,3)*cos(2*3.1415*t + ((A*x[1]*pow(sin(3.1415*t),2))/H));
    auto p5 = A*(1 + (A*x[0]*(1 - cos(2*3.1415*t))/(2*H)))/(2*H*rho*pow(2*H + A*x[0] - A*x[0]*cos(2*3.1415*t),2));
    //std::cout<< "p5="<< p5 <<std::endl;
    auto p6 = -8*pow(H,2)*lambda + 8*A*H*mu*x[0] + 3*pow(A,2)*mu*pow(x[0],2)-4*A*mu*x[0]*(2*H+A*x[0])*cos(2*3.1415*t);
    auto p7 = pow(A,2)*mu*pow(x[0],2)*cos(4*3.1415*t) + 8*pow(H,2)*lambda*log(1+((A*x[0]*pow(sin(3.1415*t),2))/H));
    auto p8 = -12*A*H*x[1]-4*pow(A,2)*x[0]*x[1] + A*(8*H + 7*A*x[0])*x[1]*cos(2*3.1415*t) + 4*A*(H - A*x[0])*x[1]*cos(4*3.1415*t);
    //std::cout<< "p8=" << p8 <<std::endl;
    auto p9 = pow(A,2)*x[0]*x[1]*cos(6*3.1415*t) + 32*pow(H,2)*sin((A*x[1]*pow(sin(3.1415*t),2))/H);
    auto p10 = -8*pow(H,2)*sin(2*3.1415*t-(A*x[1]*pow(sin(3.1415*t),2))/H) + 8*pow(H,2)*sin(2*3.1415*t + (A*x[1]*pow(sin(3.1415*t),2))/
		H);
    //std::cout<< "p10=" << p10 <<std::endl;
    auto br = pow(3.1415,2)*pow(1/sin(3.1415*t),4)*(p1+p2+p3+p4)/(32*A*pow(H,2))+p5*(p6+p7);
    auto b_theta = pow(3.1415,2)*pow(1/sin(3.1415*t),4)*(p8+p9+p10)/(8*A*H);
    //auto temp1 = pow(1/sin(3.1415*t),4);
   // auto temp2 = (32*A*pow(H,2));
    //auto temp3 = p5*(p6+p7);
    //std::cout<< "pow(3.1415,2)*pow(1/sin(3.1415*t),4)*(p1+p2+p3+p4)" << temp1 <<std::endl;
    //std::cout<< "(32*A*pow(H,2))" << temp2 <<std::endl;
    //std::cout<< "p5*(p6+p7)" << temp3 <<std::endl;
    //std::cout<< "br=" << br <<std::endl;
    //std::cout<< "b_theta=" << b_theta <<std::endl;
    auto bx = br*cos(alpha)-b_theta*sin(alpha);
    auto by = br*sin(alpha)+b_theta*cos(alpha);
    //std::cout<< "bx=" << bx <<std::endl;
    //std::cout<< "by=" << by <<std::endl;
    
   force(0)=bx;
   force(1)=by;
    
   //std::cout<< force <<std::endl;
   return force*t; */

   return 0.0*x;

  };
  solid_solver.addBodyForce(body_force); 

 //actual traction tensor
    auto traction = [E,nu,G](const auto& x, const tensor<double, dim>& N, const double) {
    using std::cos;
    using std::sin;
    using std::log;
    using std::pow;

    auto lambda=E*nu/(1+nu)/(1-2*nu);
    auto mu=G;
    //auto A=M_PI_2;
    auto H=8.0;
    double t=0.5;
    double B = 3.1415*0.5*0.5*(1.0 - cos(2.0*3.1415*t));
    auto alpha = get_value(B*x[1]/H);
    auto BL=1+(B*x[0])/H;
    tensor<double, 2, 2> U{{{1, 0},{0, get_value(BL)}}};
    auto J = det(U);
    const tensor<double, 2, 2> I=DenseIdentity<2>();
    auto sigma = (lambda*log(J)/J*I)+mu/J*(dot(U,U)-I);
    std::cout<< sigma << std::endl;
    const tensor<double, 2, 2> Q{{{cos(alpha), -sin(alpha)},{sin(alpha), cos(alpha)}}};

    //evaluate tractions on each face of the bar
    return dot(Q,dot(sigma,N)); 
   
     
  };
  solid_solver.setPiolaTraction(traction);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  double dt = 0.1;
  solid_solver.advanceTimestep(dt);

  // Output the sidre-based plot files
  solid_solver.outputState("visit_output");


  auto exact_disp = [](const mfem::Vector& X, mfem::Vector& u) {
    // u = x - X, where x = 2*X + 0*Y + 0*Z
    u[0] = X[0];
    u[1] = 0;
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

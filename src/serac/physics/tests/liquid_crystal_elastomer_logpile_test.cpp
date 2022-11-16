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
#include "serac/physics/solid_mechanics.hpp"
#include "serac/physics/materials/liquid_crystal_elastomer.hpp"

using namespace serac;

using serac::solid_mechanics::default_static_options;

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  axom::slic::SimpleLogger logger;

  constexpr int p = 1;
  constexpr int dim = 3;
  
  int num_steps = 11;

  int serial_refinement   = 0;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "LCE_logpile_test");

  // Construct the appropriate dimension mesh and give it to the data store 
  
  // std::string filename = SERAC_REPO_DIR "/data/meshes/LCE_logpile_mesh_noPlates.g";
  std::string filename = SERAC_REPO_DIR "/data/meshes/LCE_finalLogMesh_2layers_coarse.g";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  double initial_temperature = 290; // 270.0; //300.0;
  double final_temperature =  400; // 380.0; // 430.0;
  FiniteElementState temperature(
      StateManager::newState(FiniteElementState::Options{.order = p, .name = "temperature"}));

  temperature = initial_temperature;

  auto fec = std::unique_ptr< mfem::FiniteElementCollection >(new mfem::L2_FECollection(p, dim));

  FiniteElementState gamma(
      StateManager::newState(FiniteElementState::Options{.order = p, .coll = std::move(fec), .name = "gamma"}));

  // orient fibers vary based on provided function:
  //
  //      ━━━━━━━━━━━━━━━━━━━━━━━━━━
  // y   /                         /┃
  // ^  /                      4  / ┃
  // | /                       | /  ┃
  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━┓-- 4┃
  // ┃ | | | | | | | | | | | | ┃    ┃
  // ┃ | | | | | | | | | | | | ┃    ┃
  // ┃ | | | | | | | | | | | | ┃    ┃
  // ┃ | | | | | | | | | | | | ┃    ┃
  // ┃ | | | | | | | | | | | | ┃    ┃
  // ┃ - - - - - - - - - - - - ┃    ┃
  // ┃ - - - - - - - - - - - - ┃    ┃
  // ┃ - - - - - - - - - - - - ┃   /
  // ┃ - - - - - - - - - - - - ┃  /
  // ┃ - - - - - - - - - - - - ┃ /
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━┛--> x

  int lceArrangementTag = 2;
  auto gamma_func = [lceArrangementTag](const mfem::Vector& x, double) -> double 
  {
    if (lceArrangementTag==1)
    {
      return  M_PI_2;
    }
    else if (lceArrangementTag==2)
    {
      // r = 0.125
      return  M_PI_2;
    }
    else if (lceArrangementTag==2)
    {
      // Gyroid
      double a = 4;
      double LSF = sin(2*M_PI/a*x[0])*cos(2*M_PI/a*x[1]) 
                + sin(2*M_PI/a*x[1])*cos(2*M_PI/a*x[2]) 
                + sin(2*M_PI/a*x[2])*cos(2*M_PI/a*x[0]);

      return (LSF > 0.0) ? 0.667*M_PI_2 : 0.333*M_PI_2; 
    }
    else if (lceArrangementTag==3)
    {
      // Straight rods
      double rad = 0.5;
      double LSF_rod_1 = std::pow(x[0]-4.0, 2) + std::pow(x[1]-4.0, 2) - std::pow(rad, 2);
      double LSF_rod_2 = std::pow(x[2]-4.0, 2) + std::pow(x[1]-4.0, 2) - std::pow(rad, 2);
      double LSF_rod_3 = std::pow(x[0]-4.0, 2) + std::pow(x[2]-4.0, 2) - std::pow(rad, 2);

      // Inclined rod
      // double rotAngle =  M_PI_2/2.0; // 0.785398; // 0.6; //

      // double xp = x[0]; //  x[0]*cos(rotAngle) - x[1]*sin(rotAngle);
      // double yp = x[1]*cos(-rotAngle) - x[2]*sin(-rotAngle); // x[0]*sin(rotAngle) + x[1]*cos(rotAngle);
      // double zp = x[1]*sin(-rotAngle) + x[2]*cos(-rotAngle); // x[2];

      // double xpp =  xp*cos(rotAngle) - yp*sin(rotAngle); // xp;
      // // double ypp = xp*sin(rotAngle) + yp*cos(rotAngle); // yp*cos(-rotAngle) - zp*sin(-rotAngle);
      // double zpp = zp; // yp*sin(-rotAngle) + zp*cos(-rotAngle);
      
      // double LSF_rod_4 = std::pow(xpp, 2) + std::pow(zpp, 2) - std::pow(rad, 2);
      
      double LSF_rod_4 = std::pow(x[0]-x[1], 2) + std::pow(x[1]-x[2], 2)+ std::pow(x[2]-x[0], 2) - 3*std::pow(rad, 2);

      // Sphere
      double LSF_sph = std::pow(x[0], 2) + std::pow(x[1], 2) + std::pow(x[2], 2) - std::pow(2.75*rad, 2);

      // Combine LSFs4d
      double final_LSF = std::min(std::min(std::min(std::min(LSF_rod_1, LSF_rod_2),LSF_rod_3),LSF_rod_4),LSF_sph);

      return (final_LSF > 0.0) ? 1.0*M_PI_2 : 0.0*M_PI_2; 
    }
    else
    {
      // Spheres (not ready yet)
      double rad = 0.65;
      return ( 
        std::pow(x[0]-3.0, 2) + std::pow(x[1]-3.0, 2) - std::pow(rad, 2) < 0.0 ||  
        std::pow(x[0]-1.0, 2) + std::pow(x[1]-3.0, 2) - std::pow(rad, 2) < 0.0 ||  
        std::pow(x[0]-3.0, 2) + std::pow(x[1]-1.0, 2) - std::pow(rad, 2) < 0.0 ||  
        std::pow(x[0]-1.0, 2) + std::pow(x[1]-1.0, 2) - std::pow(rad, 2) < 0.0
        )? 0.333*M_PI_2 : 0.667*M_PI_2; 
    }
  };

  mfem::FunctionCoefficient coef(gamma_func);
  gamma.project(coef);

  // Construct a functional-based solid mechanics solver
    SolidMechanics<p, dim, Parameters< H1<p>, L2<p> > > solid_solver(default_static_options, GeometricNonlinearities::On, "lce_solid_functional");
  // SolidMechanics<p, dim, Parameters< H1<p>, L2<p> > > solid_solver(default_static_options, GeometricNonlinearities::Off, FinalMeshOption::Reference,
  //                                      "solid_functional", {temperature, gamma});
  // SolidMechanics<p, dim, Parameters< H1<p>, L2<p> > > solid_solver(default_static_options, GeometricNonlinearities::On, FinalMeshOption::Reference,
  //                                      "solid_functional", {temperature, gamma});

  double density = 1.0;
  double E = 1.0e-1; // 1e-2
  double nu = 0.38; // 0.3; // 0.49
  double shear_modulus = 0.5*E/(1.0 + nu);
  double bulk_modulus = E / 3.0 / (1.0 - 2.0*nu);
  double order_constant = 10.5; // 10; // 8; // 15; // 10; // 6.0;
  double order_parameter = 0.95;
  double transition_temperature = 348; // 350; // 330; //  370.0;
  double Nb2 = 1.0;
  
  LiqCrystElast_Brighenti mat(density, shear_modulus, bulk_modulus, order_constant, order_parameter, transition_temperature, Nb2);

  constexpr int TEMPERATURE_INDEX = 0;
  constexpr int GAMMA_INDEX       = 1;

  solid_solver.setParameter(temperature, TEMPERATURE_INDEX);
  solid_solver.setParameter(gamma, GAMMA_INDEX);

  LiqCrystElast_Brighenti::State initial_state{};

  auto qdata = solid_solver.createQuadratureDataBuffer(initial_state);

  solid_solver.setMaterial(DependsOn<TEMPERATURE_INDEX, GAMMA_INDEX>{}, mat, qdata);

  // prescribe symmetry conditions
  // auto zeroFunc = []( const mfem::Vector /*x*/){ return 0.0;};
  // solid_solver.setDisplacementBCs({1}, zeroFunc, 0); // bottom face x-dir disp = 0
  // solid_solver.setDisplacementBCs({1}, zeroFunc, 1); // bottom face y-dir disp = 0
  // solid_solver.setDisplacementBCs({1}, zeroFunc, 2); // bottom face z-dir disp = 0

  auto bc = [](const mfem::Vector&, mfem::Vector& bc_vec) -> void { bc_vec = 0.0; };
  solid_solver.setDisplacementBCs({1}, bc); // bottom face = 0

  // auto prescDispFunc = []( const mfem::Vector /*x*/){ return -0.001;};
  // solid_solver.setDisplacementBCs({2}, prescDispFunc, 2); // bottom face z-dir disp = 0

  // solid_solver.setPiolaTraction([](auto x, auto /*n*/, auto /*t*/){
  //   return tensor<double, 3>{0, 0, -5.0e-3 * (x[2] > 0.45)};
  // });

  // solid_solver.setPiolaTraction([](const tensor<double, dim>& x, const tensor<double, dim> & n, const double) {
  //   if (x[2] > 0.45) {
  //     return -1.0e-2 * n;
  //   }
  //   return 0.0 * n;
  // });

  // solid_solver.setPiolaTraction([](auto x, auto /*n*/, auto /*t*/){
  //   return tensor<double, 3>{0, 0, -10 * (x[2] > 0.45)};
  // });

  auto zero_displacement = [](const mfem::Vector&, mfem::Vector& u) -> void { u = 0.0; };
  solid_solver.setDisplacement(zero_displacement);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  std::string output_filename = "LCE_logpile_test_paraview_90d";
  solid_solver.outputState(output_filename); 


  double t = 0.0;
  double tmax = 1.0;
  double dt = tmax / num_steps;
  
  for (int i = 0; i < num_steps; i++) 
  {
    solid_solver.advanceTimestep(dt);
    solid_solver.outputState(output_filename);

    FiniteElementState &displacement = solid_solver.displacement();
    auto &fes = displacement.space();

    mfem::Vector dispVecX(fes.GetNDofs()); dispVecX = 0.0;
    mfem::Vector dispVecY(fes.GetNDofs()); dispVecY = 0.0;
    mfem::Vector dispVecZ(fes.GetNDofs()); dispVecZ = 0.0;

    for (int k = 0; k < fes.GetNDofs(); k++) 
    {
      dispVecX(k) = displacement(3*k+0);
      dispVecY(k) = displacement(3*k+1);
      dispVecZ(k) = displacement(3*k+2);
    }

    if(rank==0)
    {
      std::cout 
      << "\n... Entering time step: "<< i + 1
      << "\n... At time: "<< t
      << "\n... And with uniform temperature of: " << initial_temperature * (1.0 - (t / tmax)) + final_temperature * (t / tmax) 
      <<"\n... Min X displacement: " << dispVecX.Min()
      <<"\n... Max X displacement: " << dispVecX.Max()
      <<"\n... Min Y displacement: " << dispVecY.Min()
      <<"\n... Max Y displacement: " << dispVecY.Max()
      <<"\n... Min Z displacement: " << dispVecZ.Min()
      <<"\n... Max Z displacement: " << dispVecZ.Max()
      << std::endl;
    }

    t += dt;
    temperature = initial_temperature * (1.0 - (t / tmax)) + final_temperature * (t / tmax);
    
    
    // if(rank==0)
    // {
    //   std::cout<<"... Max displacement = " << displacement.Max()<<std::endl;
    // }
    // if(i>0){exit(0);}
  }

  MPI_Finalize();

}

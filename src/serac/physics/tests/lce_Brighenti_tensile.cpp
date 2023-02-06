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

#define LOAD_DRIVEN
// #undef LOAD_DRIVEN

#define FULL_DOMAIN
// #undef FULL_DOMAIN

using namespace serac;

using serac::solid_mechanics::default_static_options;

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  axom::slic::SimpleLogger logger;
  axom::slic::setIsRoot(rank == 0);

  constexpr int p = 2;
  constexpr int dim = 3;
  
  int num_steps = 4;

  // Create DataStore
  axom::sidre::DataStore datastore;
#ifdef LOAD_DRIVEN
  serac::StateManager::initialize(datastore, "LCE_tensile_test_load");
#else
  serac::StateManager::initialize(datastore, "LCE_tensile_test_temp");
#endif

  // Construct the appropriate dimension mesh and give it to the data store
  int nElem = 2;
#ifdef FULL_DOMAIN
  double lx = 0.25e-3, ly = 3.0e-3, lz = 3.0e-3;
  mfem::Mesh cuboid = mfem::Mesh(mfem::Mesh::MakeCartesian3D(nElem, 4*nElem, 4*nElem, mfem::Element::HEXAHEDRON, lx, ly, lz));
#else
  double lx = 0.25e-3/2, ly = 3.0e-3, lz = 3.0e-3/2;
  mfem::Mesh cuboid = mfem::Mesh(mfem::Mesh::MakeCartesian3D(nElem, 4*nElem, 2*nElem, mfem::Element::HEXAHEDRON, lx, ly, lz));
#endif

  auto mesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, cuboid);

  serac::StateManager::setMesh(std::move(mesh));

  double initial_temperature = 25 + 273;
  double final_temperature = 430.0;
  FiniteElementState temperature(
      StateManager::newState(FiniteElementState::Options{.order = p, .name = "temperature"}));

  temperature = initial_temperature + 0.0*final_temperature;

  FiniteElementState gamma(
      StateManager::newState(FiniteElementState::Options{.order = p, .vector_dim = 1, .element_type = ElementType::L2, .name = "gamma"}));

  // orient fibers in the beam like below (horizontal when y < 0.5, vertical when y > 0.5):
  //
  // y
  // ^                         4
  // |                         |
  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━┓-- 4
  // ┃ - - - - - - - - - - - - ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━┛--> x

  int lceArrangementTag = 1;
  auto gamma_func = [lceArrangementTag](const mfem::Vector& x, double) -> double 
  { 
    if (lceArrangementTag==1)
    {
      return M_PI_2;
    }
    else if (lceArrangementTag==2)
    {
      return (x[1] > 2.0) ? M_PI_2 : 0.0; 
    }
    else if (lceArrangementTag==3)
    {
      return ( (x[0]-2.0)*(x[1]-2.0) > 0.0) ? 0.333*M_PI_2 : 0.667*M_PI_2; 
    }
    else
    {
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
  IterativeSolverOptions default_linear_options = {.rel_tol     = 1.0e-6,
                                                       .abs_tol     = 1.0e-16,
                                                       .print_level = 0,
                                                       .max_iter    = 600,
                                                       .lin_solver  = LinearSolver::GMRES,
                                                       .prec        = HypreBoomerAMGPrec{}};
  NonlinearSolverOptions default_nonlinear_options = {
    .rel_tol = 1.0e-4, .abs_tol = 1.0e-7, .max_iter = 6, .print_level = 1};
  SolidMechanics<p, dim, Parameters< H1<p>, L2<p> > > solid_solver({default_linear_options, default_nonlinear_options}, GeometricNonlinearities::Off,
                                       "lce_solid_functional");

  constexpr int TEMPERATURE_INDEX = 0;
  constexpr int GAMMA_INDEX       = 1;

  solid_solver.setParameter(TEMPERATURE_INDEX, temperature);
  solid_solver.setParameter(GAMMA_INDEX, gamma);

  double density = 1.0;
  double E = 5.0e7;
  double nu = 0.45;
  double shear_modulus = 0.5*E/(1.0 + nu);
  double bulk_modulus = E / 3.0 / (1.0 - 2.0*nu);
  double order_constant = 10; 
  double order_parameter = 0.20;
  double transition_temperature = 348;
  double Nb2 = 1.0;

  LiqCrystElast_Brighenti mat(density, shear_modulus, bulk_modulus, order_constant, order_parameter, transition_temperature, Nb2);

  LiqCrystElast_Brighenti::State initial_state{};

  auto qdata = solid_solver.createQuadratureDataBuffer(initial_state);

  solid_solver.setMaterial(DependsOn<TEMPERATURE_INDEX, GAMMA_INDEX>{}, mat, qdata);

#ifdef FULL_DOMAIN
  // Fixed bottom
  solid_solver.setDisplacementBCs({2}, [](const mfem::Vector&, mfem::Vector& u) -> void { u = 0.0; });
#else
  // prescribe symmetry conditions
  auto zeroFunc = [](const mfem::Vector /*x*/){ return 0.0;};
  solid_solver.setDisplacementBCs({1}, zeroFunc, 2); // bottom face y-dir disp = 0
  solid_solver.setDisplacementBCs({2}, zeroFunc, 1); // left face x-dir disp = 0
  solid_solver.setDisplacementBCs({5}, zeroFunc, 0); // back face z-dir disp = 0
#endif

#ifdef LOAD_DRIVEN
  auto ini_displacement = [](const mfem::Vector&, mfem::Vector& u) -> void { u = 0.0000001; };

  double iniLoadVal = 1.0e0;
#ifdef FULL_DOMAIN
  double maxLoadVal = 4*1.3e0/lx/lz;
#else
  double maxLoadVal = 1.3e0/lx/lz;
#endif
  double loadVal = iniLoadVal + 0.0 * maxLoadVal;
  solid_solver.setPiolaTraction([&loadVal, ly](auto x, auto /*n*/, auto /*t*/){

    return tensor<double, 3>{0, loadVal * (x[1]>0.99*ly), 0};
  });

#else
  auto ini_displacement = [](const mfem::Vector&, mfem::Vector& u) -> void { u = 0.0; };
#endif
  solid_solver.setDisplacement(ini_displacement);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
#ifdef LOAD_DRIVEN
  std::string output_filename = "sol_lce_tensile_load";
#else
  std::string output_filename = "sol_lce_tensile_temp";
#endif
  solid_solver.outputState(output_filename); 


  // QoI for output:
  auto& pmesh = serac::StateManager::mesh();
  Functional<double(H1<p, dim>)> avgYDispQoI({&solid_solver.displacement().space()});
  avgYDispQoI.AddSurfaceIntegral(
      DependsOn<0>{},
      [=](auto x, auto n, auto displacement) {
        auto [u, du_dxi] = displacement;
        return dot(u, n) * ((x[1] > 0.99 * ly) ? 1.0 : 0.0);
      },
      pmesh);

  Functional<double(H1<p, dim>)> area({&solid_solver.displacement().space()});
  area.AddSurfaceIntegral(
      DependsOn<>{}, [=](auto x, auto /*n*/) { return (x[1] > 0.99 * ly) ? 1.0 : 0.0; }, pmesh); 

  double initial_area = area(solid_solver.displacement());
  if(rank==0)
  {
    std::cout << "... Initial Area of the top surface: " << initial_area << std::endl;
  }

  double t = 0.0;
  double tmax = 1.0;
  double dt = tmax / num_steps;
  double gblDispYmax;
  bool outputDispInfo(true);

  for (int i = 0; i < (num_steps+1); i++) 
  {
    if(rank==0)
    {
      std::cout 
      << "\n\n............................"
      << "\n... Entering time step: "<< i + 1
      << "\n............................\n"
      << "\n... At time: "<< t
#ifdef LOAD_DRIVEN
      << "\n... And with a tension load of: " << loadVal <<" ("<<loadVal/maxLoadVal*100<<"\% of max)"
      << "\n... And with uniform temperature of: " << initial_temperature
#else
      << "\n... And with uniform temperature of: " << initial_temperature * (1.0 - (t / tmax)) + final_temperature * (t / tmax) 
#endif
      << std::endl;
    }
    
    solid_solver.advanceTimestep(dt);
    solid_solver.outputState(output_filename);

    double current_qoi = avgYDispQoI(solid_solver.displacement());
    double current_area = area(solid_solver.displacement());

    if(outputDispInfo)
    {
      // FiniteElementState &displacement = solid_solver.displacement();
      auto &fes = solid_solver.displacement().space();
      mfem::ParGridFunction displacement_gf = solid_solver.displacement().gridFunction();
      mfem::Vector dispVecY(fes.GetNDofs()); dispVecY = 0.0;

      for (int k = 0; k < fes.GetNDofs(); k++) 
      {
        dispVecY(k) = displacement_gf(3*k+1);
      }

      double lclDispYmax = dispVecY.Max();
      MPI_Allreduce(&lclDispYmax, &gblDispYmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

      if(rank==0)
      {
        std::cout 
        <<"\n... Max Y displacement: " << gblDispYmax
        <<"\n... The QoIVal is: " << current_qoi
        <<"\n... The top surface current area is: " << std::setprecision(9) << current_area
        <<"\n... The vertical displacement integrated over the top surface is: " << current_qoi/current_area
        << std::endl;
      }

      if(std::isnan(gblDispYmax))
      {
        if(rank==0)
        {
          std::cout << "... Solution blew up... Check boundary and initial conditions." << std::endl;
        }
        exit(1);
      }
    }

    t += dt;
#ifdef LOAD_DRIVEN
    loadVal = iniLoadVal  + (maxLoadVal - iniLoadVal) * std::pow( t / tmax, 0.75  );
#else
    temperature = initial_temperature * (1.0 - (t / tmax)) + final_temperature * (t / tmax);
#endif
  }

  EXPECT_NEAR(gblDispYmax, 0.000202917533, 1.0e-8);

  MPI_Finalize();

}
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
// #include "serac/physics/materials/liquid_crystal_elastomer_material.hpp"

#define LOAD_DRIVEN
// #undef LOAD_DRIVEN

// #define USER_MESH
#undef USER_MESH

using namespace serac;

using serac::solid_mechanics::default_static_options;

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  axom::slic::SimpleLogger logger;

  constexpr int p = 1;
  constexpr int dim = 3;
  
  int num_steps = 10;

  // Create DataStore
  axom::sidre::DataStore datastore;
#ifdef LOAD_DRIVEN
  serac::StateManager::initialize(datastore, "LCE_tensile_test_load");
#else
  serac::StateManager::initialize(datastore, "LCE_tensile_test_temp");
#endif

  // Construct the appropriate dimension mesh and give it to the data store

#ifdef USER_MESH
  int serial_refinement   = 0;
  int parallel_refinement = 1;

  // std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex-flat.mesh";
  std::string filename = SERAC_REPO_DIR "/data/meshes/LCE_tensileTestSpecimen_nonDim.g"; 
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
#else
  int nElem = 4;
  ::mfem::Mesh cuboid = mfem::Mesh(mfem::Mesh::MakeCartesian3D(2*nElem, 10*nElem, nElem, mfem::Element::HEXAHEDRON, 0.67/2, 10.0/2, 0.25/2));
  auto mesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, cuboid);
#endif

  serac::StateManager::setMesh(std::move(mesh));

  double initial_temperature = 25 + 273;// 300.0;
  double final_temperature = 430.0;
  FiniteElementState temperature(
      StateManager::newState(FiniteElementState::Options{.order = p, .name = "temperature"}));

  temperature = initial_temperature + 0.0*final_temperature;

  auto fec = std::unique_ptr< mfem::FiniteElementCollection >(new mfem::L2_FECollection(p, dim));

  FiniteElementState gamma(
      StateManager::newState(FiniteElementState::Options{.order = p, .coll = std::move(fec), .name = "gamma"}));

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
      // return (x[0] > 1.0) ? M_PI_2 : 0.0; 
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
                                                       .max_iter    = 500,
                                                       .lin_solver  = LinearSolver::GMRES,
                                                       .prec        = HypreBoomerAMGPrec{}};
  NonlinearSolverOptions default_nonlinear_options = {
    .rel_tol = 1.0e-4, .abs_tol = 1.0e-7, .max_iter = 5, .print_level = 1};
SolidMechanics<p, dim, Parameters< H1<p>, L2<p> > > solid_solver({default_linear_options, default_nonlinear_options}, GeometricNonlinearities::Off,
                                       "lce_solid_functional");
  // SolidMechanics<p, dim, Parameters< H1<p>, L2<p> > > solid_solver(solid_mechanics::default_static_options, GeometricNonlinearities::Off,
  //                                      "lce_solid_functional");

  constexpr int TEMPERATURE_INDEX = 0;
  constexpr int GAMMA_INDEX       = 1;

  solid_solver.setParameter(temperature, TEMPERATURE_INDEX);
  solid_solver.setParameter(gamma, GAMMA_INDEX);

  double density = 1.0;
  double E = 1.0;
  double nu = 0.49;
  double shear_modulus = 0.5*E/(1.0 + nu);
  double bulk_modulus = E / 3.0 / (1.0 - 2.0*nu);
  double order_constant = 6.0;
  double order_parameter = 0.7;
  double transition_temperature = 370.0;
  double Nb2 = 1.0;
  
  LiquidCrystalElastomer mat(density, shear_modulus, bulk_modulus, order_constant, order_parameter, transition_temperature, Nb2);

  LiquidCrystalElastomer::State initial_state{};

  auto qdata = solid_solver.createQuadratureDataBuffer(initial_state);

  solid_solver.setMaterial(DependsOn<TEMPERATURE_INDEX, GAMMA_INDEX>{}, mat, qdata);

  // prescribe symmetry conditions
  auto zeroFunc = [](const mfem::Vector /*x*/){ return 0.0;};
  solid_solver.setDisplacementBCs({1}, zeroFunc, 2); // bottom face y-dir disp = 0
  solid_solver.setDisplacementBCs({2}, zeroFunc, 1); // left face x-dir disp = 0
  // solid_solver.setDisplacementBCs({2}, [](const mfem::Vector&, mfem::Vector& u) -> void { u = 0.0; });
  solid_solver.setDisplacementBCs({5}, zeroFunc, 0); // back face z-dir disp = 0

#ifdef LOAD_DRIVEN
  auto ini_displacement = [](const mfem::Vector&, mfem::Vector& u) -> void { u = 0.00000001; };

  double iniLoadVal = 8.0e-5;
  double loadVal = iniLoadVal;
  solid_solver.setPiolaTraction([&loadVal](auto x, auto /*n*/, auto /*t*/){

    return tensor<double, 3>{0, loadVal * (x[1]>4.95), 0};
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

  double t = 0.0;
  double tmax = 1.0;
  double dt = tmax / num_steps;
  bool outputDispInfo(true);

  for (int i = 0; i < num_steps; i++) 
  {
    if(rank==0)
    {
      std::cout 
      << "\n... Entering time step: "<< i + 1
      << "\n... At time: "<< t
#ifdef LOAD_DRIVEN
      << "\n... And with a tension load of: " << loadVal
#else
      << "\n... And with uniform temperature of: " << initial_temperature * (1.0 - (t / tmax)) + final_temperature * (t / tmax) 
#endif
      << std::endl;
    }
    
    solid_solver.advanceTimestep(dt);
    solid_solver.outputState(output_filename);

    if(outputDispInfo)
    {
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

      double gblDispXmin, lclDispXmin = dispVecX.Min();
      double gblDispXmax, lclDispXmax = dispVecX.Max();
      double gblDispYmin, lclDispYmin = dispVecY.Min();
      double gblDispYmax, lclDispYmax = dispVecY.Max();
      double gblDispZmin, lclDispZmin = dispVecZ.Min();
      double gblDispZmax, lclDispZmax = dispVecZ.Max();

      MPI_Allreduce(&lclDispXmin, &gblDispXmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&lclDispXmax, &gblDispXmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&lclDispYmin, &gblDispYmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&lclDispYmax, &gblDispYmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&lclDispZmin, &gblDispZmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&lclDispZmax, &gblDispZmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

      if(rank==0)
      {
        std::cout 
        <<"\n... Min X displacement: " << gblDispXmin
        <<"\n... Max X displacement: " << gblDispXmax
        <<"\n... Min Y displacement: " << gblDispYmin
        <<"\n... Max Y displacement: " << gblDispYmax
        <<"\n... Min Z displacement: " << gblDispZmin
        <<"\n... Max Z displacement: " << gblDispZmax
        << std::endl;
      }
    }

    t += dt;
#ifdef LOAD_DRIVEN
    loadVal = loadVal * ( 5.5 * t / tmax);
#else
    temperature = initial_temperature * (1.0 - (t / tmax)) + final_temperature * (t / tmax);
#endif
  }

  MPI_Finalize();

}

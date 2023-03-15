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
#include "serac/physics/solid_mechanics.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/physics/materials/parameterized_solid_material.hpp"

#define LOAD_DRIVEN
// #undef LOAD_DRIVEN

#define FULL_DOMAIN
// #undef FULL_DOMAIN

using namespace serac;

using serac::solid_mechanics::default_static_options;

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  axom::slic::SimpleLogger logger;
  axom::slic::setIsRoot(rank == 0);

  constexpr int p                   = 1;
  constexpr int dim                 = 3;

  // Create DataStore
  axom::sidre::DataStore datastore;
#ifdef LOAD_DRIVEN
  serac::StateManager::initialize(datastore, "mmp_tensile_test_load");
#else
  serac::StateManager::initialize(datastore, "mmp_tensile_test_disp");
#endif

  // Construct the appropriate dimension mesh and give it to the data store
  int nElem = 4;
  double lx = 9.53e-3, ly = 3.18e-3, lz = 2.e-3;
  // double lx = 0.5e-3, ly = 0.1e-3, lz = 0.05e-3;
#ifdef FULL_DOMAIN
  ::mfem::Mesh cuboid = mfem::Mesh(mfem::Mesh::MakeCartesian3D(10*nElem, 3*nElem, 2*nElem, mfem::Element::HEXAHEDRON, lx, ly, lz));
#else
  ly *= 0.5;
  lz *= 0.5;
  ::mfem::Mesh cuboid = mfem::Mesh(mfem::Mesh::MakeCartesian3D(2*nElem, 2*nElem, 1, mfem::Element::HEXAHEDRON, lx, ly, lz));
#endif
  auto mesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, cuboid);
  serac::StateManager::setMesh(std::move(mesh));

  // orient fibers in the beam like below (horizontal when y < 0.5, vertical when y > 0.5):
  //
  // y
  //
  // ^                                             8
  // |                                             |
  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓-- 1
  // ┃ | | | | | | | | | | | | | | | | | | | | | | ┃
  // ┃ - - - - - - - - - - - - - - - - - - - - - - ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛--> x

  // Construct a functional-based solid mechanics solver
  IterativeSolverOptions default_linear_options = {.rel_tol     = 1.0e-6,
                                                       .abs_tol     = 1.0e-16,
                                                       .print_level = 0,
                                                       .max_iter    = 500,
                                                       .lin_solver  = LinearSolver::GMRES,
                                                       .prec        = HypreBoomerAMGPrec{}};
  NonlinearSolverOptions default_nonlinear_options = {
    .rel_tol = 1.0e-6, .abs_tol = 1.0e-14, .max_iter = 10, .print_level = 1};
  // SolidMechanics<p, dim, Parameters< H1<p>, H1<p> > > solid_solver({default_linear_options, default_nonlinear_options}, GeometricNonlinearities::Off,
  //                                      "mmp_solid_functional");
    SolidMechanics<p, dim, Parameters< L2<p>, L2<p> > > solid_solver({default_linear_options, default_nonlinear_options}, GeometricNonlinearities::Off,
                                       "mmp_solid_functional");

  // Material properties
  double density = 1.0;
  double possionRat = 0.3; // 0.49;
  double maxElasticityParam = 500.0;
  double minElasticityParam = 1.0;

  // Parameter 1
  FiniteElementState EmodParam(StateManager::newState(FiniteElementState::Options{.order = p, .element_type = ElementType::L2, .name = "EmodParam"}));
  int problemTag = 1;
  auto EmodFunc = [problemTag,lx,minElasticityParam,maxElasticityParam](const mfem::Vector& x, double) -> double 
  {
    double Emod = 1.0;
    if (problemTag==1)
    {
      if(x[0]>lx/2.0)
      {
        Emod = maxElasticityParam;
      }
      else
      {
        Emod = minElasticityParam;
      }
    }
    else if (problemTag==2)
    {
      if(x[0]>lx/2.0)
      {
        Emod = minElasticityParam + (maxElasticityParam-minElasticityParam) * 2.0 * (x[0]/lx-0.5);
      }
      else
      {
        Emod = minElasticityParam;
      }
    }

    return Emod; 
  };
  mfem::FunctionCoefficient EmodCoef(EmodFunc);
  EmodParam.project(EmodCoef);

    // Parameter 2
  FiniteElementState GmodParam(StateManager::newState(FiniteElementState::Options{.order = p, .element_type = ElementType::L2, .name = "GmodParam"}));
  auto GmodFunc = [problemTag,lx,minElasticityParam,maxElasticityParam,possionRat](const mfem::Vector& x, double) -> double 
  {
    double Emod = 1.0;
    if (problemTag==1)
    {
      if(x[0]>lx/2.0)
      {
        Emod = maxElasticityParam;
      }
      else
      {
        Emod = minElasticityParam;
      }
    }
    else if (problemTag==2)
    {
      if(x[0]>lx/2.0)
      {
        Emod = minElasticityParam + maxElasticityParam * (x[0]-lx/2.0)/lx/2.0;
      }
      else
      {
        Emod = minElasticityParam;
      }
    }

    double shearModulus = 0.5*Emod/(1.0 + possionRat);

    return shearModulus;
  };

  mfem::FunctionCoefficient GmodCoef(GmodFunc);
  GmodParam.project(GmodCoef);

  // Set parameters
  constexpr int ELASTICITY_MODULUS_INDEX = 0;
  constexpr int SHEAR_MODULUS_INDEX = 1;
  solid_solver.setParameter(ELASTICITY_MODULUS_INDEX, EmodParam);
  solid_solver.setParameter(SHEAR_MODULUS_INDEX, GmodParam);

  // Set material
  solid_mechanics::ParameterizedNeoHookeanSolid<dim> neoMaterial{density, maxElasticityParam, minElasticityParam};
  solid_solver.setMaterial(DependsOn<ELASTICITY_MODULUS_INDEX, SHEAR_MODULUS_INDEX>{}, neoMaterial);

  // Boundary conditions:
#ifdef FULL_DOMAIN
  // Fixed bottom
  solid_solver.setDisplacementBCs({5}, [](const mfem::Vector&, mfem::Vector& u) -> void { u = 0.0; });
#else
  // Prescribe zero displacement at the supported end of the beam
  auto zero_displacement = [](const mfem::Vector& /*x*/){ return 0.0;};
  solid_solver.setDisplacementBCs({1}, zero_displacement, 1); // bottom face y-dir disp = 0
  solid_solver.setDisplacementBCs({2}, zero_displacement, 0); // left face x-dir disp = 0
  solid_solver.setDisplacementBCs({3}, zero_displacement, 2); // back face z-dir disp = 0
#endif

#ifdef LOAD_DRIVEN

  auto ini_displacement = [](const mfem::Vector&, mfem::Vector& u) -> void { u = 0.0000005; };
  double iniLoadVal = 1.0e-4;

  // double maxLoadVal = 0.25 * 1.0e1 /ly/lz ; // 2.0e6*ly*lz/4.0; // 1.5e0; // 6.36e-6
  double maxLoadVal = 0.25 * 1.0e-5 /ly/lz ; // 2.0e6*ly*lz/4.0; // 1.5e0; // 6.36e-6

#ifdef FULL_DOMAIN
  maxLoadVal *= 4;
#endif

  double loadVal = iniLoadVal + 0.0 * maxLoadVal;
  solid_solver.setPiolaTraction([&loadVal, lx](auto x, auto /*n*/, auto /*t*/){
    return tensor<double, 3>{loadVal * (x[0]>0.98*lx), 0, 0};
  });

#else

  auto ini_displacement = [](const mfem::Vector&, mfem::Vector& u) -> void { u = 0.0; };

#endif

  solid_solver.setDisplacement(ini_displacement);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  int num_steps = 50;
  
#ifdef LOAD_DRIVEN
  std::string outputFilename = "sol_mmp_tensile_load";
#else
  std::string outputFilename = "sol_mmp_tensile_disp";
#endif
  solid_solver.outputState(outputFilename);
 
  double t    = 0.0;
  double tmax = 1.0;
  double dt   = tmax / num_steps;
  bool outputDispInfo(true);

  for (int i = 0; i < num_steps; i++) 
  {

    t += dt;
#ifdef LOAD_DRIVEN
    // loadVal = iniLoadVal +  t / tmax * (maxLoadVal - iniLoadVal);
    // loadVal = iniLoadVal * std::exp( std::log(maxLoadVal/iniLoadVal) * t / tmax  );
    loadVal = iniLoadVal  + (maxLoadVal - iniLoadVal) * std::pow( t / tmax, 3.0  ); // 0.75
#else
    // elasticityParam = minElasticityParam * (tmax - t) / tmax;
#endif
    if(rank==0)
    {
      std::cout 
      << "\n\n............................"
      << "\n... Entering time step: "<< i + 1 << " (/" << num_steps << ")"
      << "\n............................\n"
#ifdef LOAD_DRIVEN
      << "\n... Using a tension load of: " << loadVal <<" ("<<loadVal/maxLoadVal*100<<"\% of max)"
      << "\n... With max tension load of: " << maxLoadVal
#else
      << "\n... Using order parameter: "<< minElasticityParam * (tmax - t) / tmax
#endif
      << std::endl;
    }

    solid_solver.advanceTimestep(dt);
    solid_solver.outputState(outputFilename);

//     auto [K, K_e] = solid_solver.stiffnessMatrix();
//     K.Print("Kmat");
//     K_e.Print("K_e_mat");
    
// exit(0);
    if(outputDispInfo)
    {
      // FiniteElementState &displacement = solid_solver.displacement();
      auto &fes = solid_solver.displacement().space();
      mfem::ParGridFunction displacement_gf = solid_solver.displacement().gridFunction();
      mfem::Vector dispVecX(fes.GetNDofs()); dispVecX = 0.0;
      mfem::Vector dispVecY(fes.GetNDofs()); dispVecY = 0.0;
      mfem::Vector dispVecZ(fes.GetNDofs()); dispVecZ = 0.0;

      for (int k = 0; k < fes.GetNDofs(); k++) 
      {
        dispVecX(k) = displacement_gf(3*k+0);
        dispVecY(k) = displacement_gf(3*k+1);
        dispVecZ(k) = displacement_gf(3*k+2);
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
        <<"\n... Min Y displacement: " << gblDispYmin
        <<"\n... Min Z displacement: " << gblDispZmin 

        <<"\n\n... Max X displacement: " << gblDispXmax
        <<"\n... Max Y displacement: " << gblDispYmax
        <<"\n... Max Z displacement: " << gblDispZmax 
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
  }

  MPI_Finalize();
}

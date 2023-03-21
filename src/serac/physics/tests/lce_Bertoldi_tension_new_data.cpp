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

// #define FULL_DOMAIN
#undef FULL_DOMAIN

// #define NEMATIC_STATE 
#undef NEMATIC_STATE

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
  serac::StateManager::initialize(datastore, "lce_tensile_test_load_bertoldi_new_data");
#else
  serac::StateManager::initialize(datastore, "lce_tensile_test_disp_bertoldi_new_data");
#endif

  // Construct the appropriate dimension mesh and give it to the data store
  int nElem = 4;

#ifdef NEMATIC_STATE
  double lx = 2.542e-3, ly = 20.26e-3, lz = 0.149e-3;
  // double lx = 2.5e-3, ly = 3.0e-3, lz = 2.1e-3;
#else
  double lx = 3.547e-3, ly = 16.67e-3, lz = 0.129e-3;
#endif

#ifdef FULL_DOMAIN
  ::mfem::Mesh cuboid = mfem::Mesh(mfem::Mesh::MakeCartesian3D(12*nElem, 22*nElem, nElem, mfem::Element::HEXAHEDRON, lx, ly, lz));
  // ::mfem::Mesh cuboid = mfem::Mesh(mfem::Mesh::MakeCartesian3D(12, 40, 5 + 0*nElem, mfem::Element::HEXAHEDRON, lx, ly, lz));
  // ::mfem::Mesh cuboid = mfem::Mesh(mfem::Mesh::MakeCartesian3D(10, 50, 4 + 0*nElem, mfem::Element::HEXAHEDRON, lx, ly, lz));
#else
  lx *= 0.5;
  ly *= 0.5;
  lz *= 0.5;

#ifdef NEMATIC_STATE
  // ::mfem::Mesh cuboid = mfem::Mesh(mfem::Mesh::MakeCartesian3D(8*nElem, 70*nElem, nElem, mfem::Element::HEXAHEDRON, lx, ly, lz));
  // ::mfem::Mesh cuboid = mfem::Mesh(mfem::Mesh::MakeCartesian3D(10*nElem, 70*nElem, nElem, mfem::Element::HEXAHEDRON, lx, ly, lz));
  // ::mfem::Mesh cuboid = mfem::Mesh(mfem::Mesh::MakeCartesian3D(5*nElem, 50*nElem, nElem, mfem::Element::HEXAHEDRON, lx, ly, lz));
  ::mfem::Mesh cuboid = mfem::Mesh(mfem::Mesh::MakeCartesian3D(4*nElem, 20*nElem, nElem, mfem::Element::HEXAHEDRON, lx, ly, lz));
#else
  ::mfem::Mesh cuboid = mfem::Mesh(mfem::Mesh::MakeCartesian3D(4*nElem, 20*nElem, nElem, mfem::Element::HEXAHEDRON, lx, ly, lz));
#endif

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
  IterativeSolverOptions default_linear_options = {.rel_tol     = 1.0e-7,
                                                       .abs_tol     = 1.0e-16,
                                                       .print_level = 0,
                                                       .max_iter    = 600,
                                                       .lin_solver  = LinearSolver::GMRES,
                                                       .prec        = HypreBoomerAMGPrec{}};
  NonlinearSolverOptions default_nonlinear_options = {
    .rel_tol = 1.0e-7, .abs_tol = 1.0e-13, .max_iter = 8, .print_level = 1};
  SolidMechanics<p, dim, Parameters< H1<p>, L2<p>, L2<p> > > solid_solver({default_linear_options, default_nonlinear_options}, GeometricNonlinearities::Off,
                                       "lce_solid_functional");
  // SolidMechanics<p, dim, Parameters<H1<p>, L2<p>, L2<p>>> solid_solver(default_static_options, GeometricNonlinearities::Off,
  //                                                                 "lce_solid_functional");

  // Material properties
  double density = 1.0;
  double young_modulus = 0.4e6; //0.61e6;
  double possion_ratio = 0.48;
  double beta_param = 5.2e4; // 0.041;
  double max_order_param = 0.45;
#ifdef NEMATIC_STATE
  double initial_order_param = max_order_param;
#else
  double initial_order_param =    1.0e-8;
#endif

  // Parameter 1
  FiniteElementState orderParam(StateManager::newState(FiniteElementState::Options{.order = p, .name = "orderParam"}));
  orderParam = initial_order_param;

  // Parameter 2
  FiniteElementState gammaParam(StateManager::newState(FiniteElementState::Options{.order = p, .vector_dim = 1, .element_type = ElementType::L2, .name = "gammaParam"}));

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

  int lceArrangementTag = 1;
  auto gammaFunc = [lceArrangementTag](const mfem::Vector& x, double) -> double 
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

  mfem::FunctionCoefficient gammaCoef(gammaFunc);
  gammaParam.project(gammaCoef);

  // Paremetr 3
  // auto fec2 = std::unique_ptr<mfem::FiniteElementCollection>(new mfem::L2_FECollection(p, dim));
  FiniteElementState etaParam(StateManager::newState(FiniteElementState::Options{.order = p, .vector_dim = 1, .element_type = ElementType::L2, .name = "etaParam"}));
  auto etaFunc = [](const mfem::Vector& /*x*/, double) -> double { return 0.0; };
  mfem::FunctionCoefficient etaCoef(etaFunc);
  etaParam.project(etaCoef);

  // Set parameters
  constexpr int ORDER_INDEX = 0;
  constexpr int GAMMA_INDEX = 1;
  constexpr int ETA_INDEX   = 2;

  solid_solver.setParameter(ORDER_INDEX, orderParam);
  solid_solver.setParameter(GAMMA_INDEX, gammaParam);
  solid_solver.setParameter(ETA_INDEX, etaParam);

  // Set material
  LiqCrystElast_Bertoldi lceMat(density, young_modulus, possion_ratio, max_order_param, beta_param);
  LiqCrystElast_Bertoldi::State initial_state{};

  auto param_data = solid_solver.createQuadratureDataBuffer(initial_state);
  solid_solver.setMaterial(DependsOn<ORDER_INDEX, GAMMA_INDEX, ETA_INDEX>{}, lceMat, param_data);

  // Boundary conditions:
#ifdef FULL_DOMAIN
  // Fixed bottom
  solid_solver.setDisplacementBCs({2}, [](const mfem::Vector&, mfem::Vector& u) -> void { u = 0.0; });
#else
  // Prescribe zero displacement at the supported end of the beam
  auto zero_displacement = [](const mfem::Vector& /*x*/){ return 0.0;};
  solid_solver.setDisplacementBCs({1}, zero_displacement, 2); // bottom face y-dir disp = 0
  solid_solver.setDisplacementBCs({2}, zero_displacement, 1); // left face x-dir disp = 0
  solid_solver.setDisplacementBCs({5}, zero_displacement, 0); // back face z-dir disp = 0
#endif

#ifdef LOAD_DRIVEN

  auto ini_displacement = [](const mfem::Vector&, mfem::Vector& u) -> void { u = 1.0e-14; };
  double iniLoadVal = 2.0e-4/lx/lz;
#ifdef NEMATIC_STATE
  double maxLoadVal = 1.1064e0/lx/lz/4.0;
  // double maxLoadVal = 1.38e-1/lx/lz/4.0;
  // double maxLoadVal = 5.53e-2/lx/lz/4.0;
#else
double maxLoadVal = 1.95875e-1/lx/lz/4.0/4.0;
  // double maxLoadVal = 1.95875e-1/lx/lz/4.0;
#endif


#ifdef FULL_DOMAIN
  maxLoadVal *= 4;
#endif

  double loadVal = iniLoadVal + 0.0 * maxLoadVal;
  solid_solver.setPiolaTraction([&loadVal, ly](auto x, auto /*n*/, auto /*t*/){
    return tensor<double, 3>{0, loadVal * (x[1]>0.975*ly), 0};
  });

#else

  auto ini_displacement = [](const mfem::Vector&, mfem::Vector& u) -> void { u = 0.0; };

#endif

  solid_solver.setDisplacement(ini_displacement);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  int num_steps = 30;
  
#ifdef LOAD_DRIVEN
  std::string outputFilename = "sol_lce_bertoldi_tensile_load_new_data";
#else
  std::string outputFilename = "sol_lce_bertoldi_tensile_disp_new_data";
#endif
  solid_solver.outputState(outputFilename);
 
  double t    = 0.0;
  double tmax = 1.0;
  double dt   = tmax / num_steps;
  bool outputDispInfo(true);

  for (int i = 0; i < num_steps; i++) 
  {
        if(rank==0)
    {
      std::cout 
      << "\n\n............................"
      << "\n... Entering time step: "<< i + 1 << " (/" << num_steps << ")"
      << "\n............................"
#ifdef LOAD_DRIVEN
      << "\n... Using a tension load of: " << loadVal <<" ("<<loadVal/maxLoadVal*100<<"\% of max)"
      << "\n... With max tension load of: " << maxLoadVal
// #else
      << "\n... Using order parameter: "<< initial_order_param
#endif
      << std::endl << std::endl;
    }

    t += dt;
    solid_solver.advanceTimestep(dt);
    solid_solver.outputState(outputFilename);

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
        <<"\n... Max Y displacement: " << gblDispYmax << std::endl;
      }

      if(std::isnan(gblDispYmax) || gblDispYmax>1000*ly)
      {
        if(rank==0)
        {
          std::cout << "... Solution blew up... Check boundary and initial conditions." << std::endl;
        }
        exit(1);
      }
    }
    
#ifdef LOAD_DRIVEN
    // loadVal = iniLoadVal +  t / tmax * (maxLoadVal - iniLoadVal);
    // loadVal = iniLoadVal * std::exp( std::log(maxLoadVal/iniLoadVal) * t / tmax  );
    loadVal = iniLoadVal  + (maxLoadVal - iniLoadVal) * std::pow( t / tmax, 0.75  );
#else
    orderParam = initial_order_param * (tmax - t) / tmax;
#endif
  }

  MPI_Finalize();
}

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
#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/terminator.hpp"

#define LOAD_DRIVEN
// #undef LOAD_DRIVEN

// #define FULL_DOMAIN
#undef FULL_DOMAIN

static const int probTag_ = 9;
// Cases [1/5, 2/6, 3/7, 4/8/9] have a max Young's moudulus of [500, 200, 100, and 5] respectively. 
// Cases [1, 2, 3, 4] have no transition between material properties, 
// Cases [5, 6, 7, 8] have a 5 [mm] transition between material properties
// Case 9 is fully graded from one end to the other

using namespace serac;

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  axom::slic::SimpleLogger logger;
  axom::slic::setIsRoot(rank == 0);

  constexpr int p   = 1;
  constexpr int dim = 3;

  // Create DataStore
  axom::sidre::DataStore datastore;
#ifdef LOAD_DRIVEN
  serac::StateManager::initialize(datastore, "mmp_tensile_test_load");
#else
  serac::StateManager::initialize(datastore, "mmp_tensile_test_disp");
#endif

  // Construct the appropriate dimension mesh and give it to the data store
  int    nElem = 4;  // 2*3;
  double lx = 9.53e-3, ly = 3.18e-3 / 2, lz = 2.e-3 / 2;
  // double lx = 0.5e-3, ly = 0.1e-3, lz = 0.05e-3;
#ifdef FULL_DOMAIN
  ::mfem::Mesh cuboid =
      mfem::Mesh(mfem::Mesh::MakeCartesian3D(19 * nElem, 3 * nElem, 2 * nElem, mfem::Element::HEXAHEDRON, lx, ly, lz));
#else
  // ly *= 0.5;
  // lz *= 0.5;
  ::mfem::Mesh cuboid =
      mfem::Mesh(mfem::Mesh::MakeCartesian3D( 30 * nElem, 3 * nElem, 2 * nElem, mfem::Element::HEXAHEDRON, lx, ly, lz));
#endif
  auto mesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, cuboid);
  serac::StateManager::setMesh(std::move(mesh));

  // Construct a functional-based solid mechanics solver
  // IterativeSolverOptions          default_linear_options    = {.rel_tol     = 1.0e-6,
  //                                                  .abs_tol     = 1.0e-16,
  //                                                  .print_level = 0,
  //                                                  .max_iter    = 500,
  //                                                  .lin_solver  = LinearSolver::GMRES,
  //                                                  .prec        = HypreBoomerAMGPrec{}};
  LinearSolverOptions linear_options = {.linear_solver = LinearSolver::SuperLU};
  NonlinearSolverOptions nonlinear_options = {.nonlin_solver  = serac::NonlinearSolver::Newton,
                                              .relative_tol   = 1.0e-8,
                                              .absolute_tol   = 1.0e-14,
                                              .max_iterations = 10,
                                              .print_level    = 1};

  SolidMechanics<p, dim, Parameters<L2<p>, L2<p> > > solid_solver(
    nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options, GeometricNonlinearities::On, "mmp_solid_functional");

  // Material properties
  double density            = 1.0;
  double possionRat         = 0.3;
  double maxElasticityParam = 500.0;
  double minElasticityParam = 1.0;  // maxElasticityParam;

  switch (probTag_) {
    case 1:
    case 5:
    case 9:
      maxElasticityParam = 500.0;
      break;
    case 2:
    case 6:
      maxElasticityParam = 200.0;
      break;
    case 3:
    case 7:
      maxElasticityParam = 100.0;
      break;
    case 4:
    case 8:
      maxElasticityParam = 5.0;
      break;
    default:
      std::cout << "... Invalid problem Tag/Id" << std::endl;
      exit(0);
  }

  // Parameter 1
  FiniteElementState EmodParam(StateManager::newState(
      FiniteElementState::Options{.order = p, .element_type = ElementType::L2, .name = "EmodParam"}));
  auto               EmodFunc = [=](const mfem::Vector& x, double) -> double {
    double Emod = 1.0;
    if (probTag_ < 5) {
      if (x[0] > lx / 2.0) {
        Emod = maxElasticityParam;
      } else {
        Emod = minElasticityParam;
      }
    } else if (probTag_ < 9) {
      double transitionLength = 5.0e-3;
      double iniTransition    = (lx - transitionLength) / 2.0;
      double endTransition    = iniTransition + transitionLength;

      if (x[0] < iniTransition) {
        Emod = minElasticityParam;
      } else if (x[0] > iniTransition && x[0] < endTransition) {
        Emod = minElasticityParam +
               (maxElasticityParam - minElasticityParam) * ((x[0] - iniTransition) / transitionLength);
      } else {
        Emod = maxElasticityParam;
      }
    } else {
      Emod = minElasticityParam + (maxElasticityParam - minElasticityParam) * x[0] / lx;  // * 2.0 * (x[0]/lx-0.5);
    }

    return Emod;
  };
  mfem::FunctionCoefficient EmodCoef(EmodFunc);
  EmodParam.project(EmodCoef);

  // Parameter 2
  FiniteElementState GmodParam(StateManager::newState(
      FiniteElementState::Options{.order = p, .element_type = ElementType::L2, .name = "GmodParam"}));
  auto               GmodFunc = [=](const mfem::Vector& x, double) -> double {
    double Emod = 1.0;
    if (probTag_ < 5) {
      if (x[0] > lx / 2.0) {
        Emod = maxElasticityParam;
      } else {
        Emod = minElasticityParam;
      }
    } else if (probTag_ < 9) {
      double transitionLength = 5.0e-3;
      double iniTransition    = (lx - transitionLength) / 2.0;
      double endTransition    = iniTransition + transitionLength;

      if (x[0] < iniTransition) {
        Emod = minElasticityParam;
      } else if (x[0] > iniTransition && x[0] < endTransition) {
        Emod = minElasticityParam +
               (maxElasticityParam - minElasticityParam) * ((x[0] - iniTransition) / transitionLength);
      } else {
        Emod = maxElasticityParam;
      }
    } else {
      Emod = minElasticityParam + (maxElasticityParam - minElasticityParam) * x[0] / lx;  // * 2.0 * (x[0]/lx-0.5);
    }

    double shearModulus = 0.5 * Emod / (1.0 + possionRat);

    return shearModulus;
  };

  mfem::FunctionCoefficient GmodCoef(GmodFunc);
  GmodParam.project(GmodCoef);

  // Set parameters
  constexpr int ELASTICITY_MODULUS_INDEX = 0;
  constexpr int SHEAR_MODULUS_INDEX      = 1;
  solid_solver.setParameter(ELASTICITY_MODULUS_INDEX, EmodParam);
  solid_solver.setParameter(SHEAR_MODULUS_INDEX, GmodParam);

  // Set material
  solid_mechanics::ParameterizedNeoHookeanSolid<dim> neoMaterial{density, maxElasticityParam, minElasticityParam};
  solid_solver.setMaterial(DependsOn<ELASTICITY_MODULUS_INDEX, SHEAR_MODULUS_INDEX>{}, neoMaterial);

  // Boundary conditions:
#ifdef FULL_DOMAIN
  // Fixed bottom
  auto zero_displacement = [](const mfem::Vector& /*x*/) { return 0.0; };
  solid_solver.setDisplacementBCs({1}, zero_displacement, 2);  // bottom face y-dir disp = 0
  solid_solver.setDisplacementBCs({2}, zero_displacement, 1);  // back face z-dir disp = 0
  solid_solver.setDisplacementBCs({5}, [](const mfem::Vector&, mfem::Vector& u) -> void { u = 0.0; });
#else
  // Prescribe zero displacement at the supported end of the beam
  auto zero_displacement = [](const mfem::Vector& /*x*/) { return 0.0; };
  solid_solver.setDisplacementBCs({1}, zero_displacement, 2);  // bottom face y-dir disp = 0
  solid_solver.setDisplacementBCs({2}, zero_displacement, 1);  // left face x-dir disp = 0
  // solid_solver.setDisplacementBCs({3}, zero_displacement, 0);  // back face z-dir disp = 0

  auto is_at_center = [=](const mfem::Vector& x) {
    if (x(0) < 0.005 && x(0) > 0.0048 && x(1) < 0.05*ly && x(2) < 0.05*lz) {
      return true;
    }
    return false;
  };
  auto zero_vector   = [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; };
  solid_solver.setDisplacementBCs(is_at_center, zero_vector);
  // auto zero_scalar   = [](const mfem::Vector& /*x*/) { return 0.0; };
  // solid_solver.setDisplacementBCs(is_at_center, zero_scalar, 0);
#endif

#ifdef LOAD_DRIVEN

  auto   ini_displacement = [](const mfem::Vector&, mfem::Vector& u) -> void { u = 0.0000005; };
  double iniLoadVal       = 1.0e-4;

  // double maxLoadVal = 0.25 * 1.0e1 /ly/lz ; // 2.0e6*ly*lz/4.0; // 1.5e0; // 6.36e-6
  // double maxLoadVal = 0.25 * 1.0e-5 /ly/lz ; // 2.0e6*ly*lz/4.0; // 1.5e0; // 6.36e-6
  // double maxLoadVal = 0.25 * 1.5e-5 / (ly / 2) / (lz / 2);  // 2.0e6*ly*lz/4.0; // 1.5e0; // 6.36e-6
  double maxLoadVal = 0.25 * 2.5e-5 / (ly*lz); 

#ifdef FULL_DOMAIN
  maxLoadVal *= 4.0;
#endif

  double loadVal = iniLoadVal + 0.0 * maxLoadVal;
  solid_solver.setPiolaTraction([&loadVal, lx](auto x, auto /*n*/, auto /*t*/) {
#ifdef FULL_DOMAIN
    return tensor<double, 3>{loadVal * (x[0] > 0.9999 * lx), 0, 0};
#else
    if(x[0] < 0.0001 * lx)
    {
      return tensor<double, 3>{-loadVal, 0, 0};
    }
    else if(x[0] > 0.9999 * lx)
    {
      return tensor<double, 3>{loadVal, 0, 0};
    }
    else
    {
      return tensor<double, 3>{0, 0, 0};
    }
#endif
  });

#else

  auto        ini_displacement = [](const mfem::Vector&, mfem::Vector& u) -> void { u = 0.0; };

#endif

  solid_solver.setDisplacement(ini_displacement);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  int num_steps = 30;

#ifdef LOAD_DRIVEN
  std::string outputFilename = "sol_mmp_tensile_load_probId_";
#else
  std::string outputFilename   = "sol_mmp_tensile_disp_probId_";
#endif

  outputFilename += std::to_string(probTag_);

  solid_solver.outputState(outputFilename);

  double t    = 0.0;
  double tmax = 1.0;
  double dt   = tmax / num_steps;
  bool   outputDispInfo(true);

  for (int i = 0; i < num_steps; i++) {
    t += dt;
#ifdef LOAD_DRIVEN
    // loadVal = iniLoadVal +  t / tmax * (maxLoadVal - iniLoadVal);
    // loadVal = iniLoadVal * std::exp( std::log(maxLoadVal/iniLoadVal) * t / tmax  );
    loadVal = iniLoadVal + (maxLoadVal - iniLoadVal) * std::pow(t / tmax, 2.0);  // 0.75
#else
    // elasticityParam = minElasticityParam * (tmax - t) / tmax;
#endif
    if (rank == 0) {
      std::cout << "\n\n............................"
                << "\n... Entering time step: " << i + 1 << " (/" << num_steps << ")"
                << "\n............................\n"
#ifdef LOAD_DRIVEN
                << "\n... Using a tension load of: " << loadVal << " (" << loadVal / maxLoadVal * 100
                << " percent of max)"
                << "\n... With max tension load of: " << maxLoadVal
#else
                << "\n... Using order parameter: " << minElasticityParam * (tmax - t) / tmax
#endif
                << std::endl;
    }

    solid_solver.advanceTimestep(dt);
    solid_solver.outputState(outputFilename);

    //     auto [K, K_e] = solid_solver.stiffnessMatrix();
    //     K.Print("Kmat");
    //     K_e.Print("K_e_mat");

    // exit(0);
    if (outputDispInfo) {
      // FiniteElementState &displacement = solid_solver.displacement();
      auto&                 fes             = solid_solver.displacement().space();
      mfem::ParGridFunction displacement_gf = solid_solver.displacement().gridFunction();
      int                   numDofs         = fes.GetNDofs();
      mfem::Vector          dispVecX(numDofs);
      dispVecX = 0.0;
      mfem::Vector dispVecY(numDofs);
      dispVecY = 0.0;
      mfem::Vector dispVecZ(numDofs);
      dispVecZ = 0.0;

      for (int k = 0; k < fes.GetNDofs(); k++) {
        // dispVecX(k) = displacement_gf(3*k+0);
        // dispVecY(k) = displacement_gf(3*k+1);
        // dispVecZ(k) = displacement_gf(3*k+2);
        dispVecX(k) = displacement_gf(0 * numDofs + k);
        dispVecY(k) = displacement_gf(1 * numDofs + k);
        dispVecZ(k) = displacement_gf(2 * numDofs + k);
      }

      double gblDispXmin, lclDispXmin = dispVecX.Min();
      // double gblDispXmax, lclDispXmax = dispVecX.Max();
      double gblDispYmin, lclDispYmin = dispVecY.Min();
      // double gblDispYmax, lclDispYmax = dispVecY.Max();
      double gblDispZmin, lclDispZmin = dispVecZ.Min();
      // double gblDispZmax, lclDispZmax = dispVecZ.Max();

      MPI_Allreduce(&lclDispXmin, &gblDispXmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      // MPI_Allreduce(&lclDispXmax, &gblDispXmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&lclDispYmin, &gblDispYmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      // MPI_Allreduce(&lclDispYmax, &gblDispYmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&lclDispZmin, &gblDispZmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      // MPI_Allreduce(&lclDispZmax, &gblDispZmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

      if (rank == 0) {
        std::cout << "\n... In time step: " << i + 1 << " (/" << num_steps
                  << ")"
                  <<"\n... Min X displacement: " << gblDispXmin
                  // << "\n... Max X displacement: " << gblDispXmax 
                  << "\n... Min Y displacement: " << gblDispYmin
                  << "\n... Min Z displacement: "
                  << gblDispZmin
                  // <<"\n... Max Y displacement: " << gblDispYmax
                  // <<"\n... Max Z displacement: " << gblDispZmax
                  << std::endl;
      }

      if (std::isnan(gblDispXmin)) {
      // if (std::isnan(gblDispXmax)) {
        if (rank == 0) {
          std::cout << "... Solution blew up... Check boundary and initial conditions." << std::endl;
        }
        exit(1);
      }
    }
  }

  // MPI_Finalize();
  serac::exitGracefully();
}

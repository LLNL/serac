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
#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/terminator.hpp"

using namespace serac;

// #define USE_2D_MESH
#undef USE_2D_MESH

// #define ALT_ITER_SOLVER
#undef ALT_ITER_SOLVER

const static int problemID = 2;

int main(int argc, char* argv[])
{
  auto [num_procs, rank] = serac::initialize(argc, argv);

  constexpr int p = 1;
#ifdef USE_2D_MESH
  constexpr int dim = 2;
#else
  constexpr int dim      = 3;
#endif
  int serial_refinement   = 2;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_lce_functional");

  // Construct the appropriate dimension mesh and give it to the data store

#ifdef USE_2D_MESH
  std::string filename = SERAC_REPO_DIR "/data/meshes/squaredLattice3D.g";
#else
  std::string   filename = SERAC_REPO_DIR "/data/meshes/squaredLattice3D.g";
#endif

  auto initial_mesh = buildMeshFromFile(filename);
  auto mesh = mesh::refineAndDistribute(std::move(initial_mesh), serial_refinement, parallel_refinement);

  std::string mesh_tag{"mesh}"};
  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  // Construct a functional-based solid mechanics solver
  // LinearSolverOptions linear_options = {.linear_solver = LinearSolver::SuperLU};

  LinearSolverOptions linear_options = {.linear_solver  = LinearSolver::GMRES,
                                                      .preconditioner = Preconditioner::HypreAMG,
                                                      .relative_tol   = 1.0e-6,
                                                      .absolute_tol   = 1.0e-10,
                                                      .max_iterations = 500,
                                                      .print_level    = 0};

  NonlinearSolverOptions nonlinear_options = {.nonlin_solver  = serac::NonlinearSolver::Newton,
                                              .relative_tol   = 1.0e-8,
                                              .absolute_tol   = 1.0e-14,
                                              .max_iterations = 15,
                                              .print_level    = 1};
  SolidMechanics<p, dim, Parameters<L2<0>, L2<0>, L2<0> > > solid_solver(
      nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options, GeometricNonlinearities::On, 
      "lce_solid_free_swelling", mesh_tag, {"orderParam", "gammaParam", "etaParam"});

//   IterativeNonlinearSolverOptions default_nonlinear_options = {.rel_tol       = 1.0e-6,
//                                                                .abs_tol       = 1.0e-8,
//                                                                .max_iter      = 100,
//                                                                .print_level   = 1,
//                                                                .nonlin_solver = serac::NonlinearSolver::Newton};
//   // .nonlin_solver = serac::NonlinearSolver::Newton};
//   // .nonlin_solver = serac::NonlinearSolver::LBFGS};
//   // .nonlin_solver = serac::NonlinearSolver::KINFullStep};
//   //.nonlin_solver = serac::NonlinearSolver::KINBacktrackingLineSearch};
//   // .nonlin_solver = serac::NonlinearSolver::KINPicard};
//   // .nonlin_solver = serac::NonlinearSolver::KINFP};

// #ifdef ALT_ITER_SOLVER
//   auto custom_solver = std::make_unique<mfem::GMRESSolver>(MPI_COMM_WORLD);
//   custom_solver->SetRelTol(1.0e-8);
//   custom_solver->SetAbsTol(1.0e-16);
//   custom_solver->SetPrintLevel(0);
//   custom_solver->SetMaxIter(700);
//   custom_solver->SetKDim(500);

//   SolidMechanics<p, dim, Parameters<L2<0>, L2<0>, L2<0> > > solid_solver(
//       {CustomSolverOptions{custom_solver.get()}, default_nonlinear_options}, GeometricNonlinearities::Off,
//       "lce_solid_functional");
// #else
//   DirectSolverOptions linear_sol_options = {};
//   SolidMechanics<p, dim, Parameters<L2<0>, L2<0>, L2<0> > > solid_solver(
//       {linear_sol_options, default_nonlinear_options}, GeometricNonlinearities::Off, "lce_solid_functional");
// #endif

  // Material properties
  double density         = 1.0;
  double young_modulus   = 0.25e6; // 0.25e6; (multiply by 10e-3 to go from SI to [Kg/s/mm])
  double possion_ratio   = 0.48;
  double beta_param      = 5.2e4;  // 5.2e4; (multiply by 10e-3 to go from SI to [Kg/s/mm])
  double max_order_param = 0.4;
  double gamma_angle     = 0.0;
  double eta_angle       = 0.0;

  switch (problemID) {
    case 0:
      gamma_angle = 0.0;
      eta_angle   = 0.0;
      break;
    case 1:
      gamma_angle = M_PI_2;
      eta_angle   = 0.0;
      break;
    case 2:
      gamma_angle = M_PI_2;
      eta_angle   = 0.0;
      break;
    default:
      std::cout << "...... Wrong problem ID ......" << std::endl;
      exit(0);
  }

  // Parameter 1
  FiniteElementState orderParam(pmesh, L2<0>{}, "orderParam");
  orderParam = max_order_param;

  // Parameter 2
  FiniteElementState gammaParam(pmesh, L2<0>{}, "gammaParam");
  bool               heterogeneousGammaField = problemID == 2 ? true : false;
  auto               gammaFunc = [heterogeneousGammaField, gamma_angle](const mfem::Vector& x, double) -> double {
    if (heterogeneousGammaField) {
      // double d    = 5.0e-3;
      // double t    = 0.5e-3;
      double Hmax = 15.0e-3;
      // top wall
      // if (x[1] >= Hmax || x[0]<4.833333e-3 || (x[0]>5.333333e-3 && x[0]<Hmax)) {
        if (x[1] >= Hmax || x[0]<4.8e-3 || (x[0]>5.5e-3 && x[0]<Hmax)) {
        return 0.0;
      }
      // else 
      // {
      //     return M_PI_2;
      // }

      return M_PI_2;
    }
    return gamma_angle;
  };
  mfem::FunctionCoefficient gammaCoef(gammaFunc);
  gammaParam.project(gammaCoef);

  // Paremetr 3
  FiniteElementState        etaParam(pmesh, L2<0>{}, "etaParam");
  auto                      etaFunc = [eta_angle](const mfem::Vector& /*x*/, double) -> double { return eta_angle; };
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
  LiquidCrystalElastomerBertoldi        lceMat(density, young_modulus, possion_ratio, max_order_param, beta_param);

  solid_solver.setMaterial(DependsOn<ORDER_INDEX, GAMMA_INDEX, ETA_INDEX>{}, lceMat);

  auto zeroFunc = [](const mfem::Vector /*x*/) { return 0.0; };
  solid_solver.setDisplacementBCs({1}, zeroFunc, 0);  // left face x-dir disp = 0
  solid_solver.setDisplacementBCs({2}, zeroFunc, 1);  // bottom face y-dir disp = 0
#ifndef USE_2D_MESH
  solid_solver.setDisplacementBCs({5}, zeroFunc, 2);  // back face z-dir disp = 0
#endif

  double iniDispVal = 1.0e-6;
  auto ini_displacement = [iniDispVal](const mfem::Vector&, mfem::Vector& u) -> void { u = iniDispVal; };
  solid_solver.setDisplacement(ini_displacement);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  std::string outputFilename;
  switch (problemID) {
    case 0:
      outputFilename = "sol_lce_bertoldi_logpile_gamma_00_eta_00";
      break;
    case 1:
      outputFilename = "sol_lce_bertoldi_logpile_gamma_90_eta_00";
      break;
    case 2:
      outputFilename = "sol_lce_bertoldi_logpile_varying_angle_order_0p45";
      break;
    default:
      std::cout << "...... Wrong problem ID ......" << std::endl;
      exit(0);
  }
#ifdef USE_2D_MESH
  outputFilename += "_2D";
#endif

#ifdef USE_2X1_LATTICE
  outputFilename = "sol_lce_bertoldi_logpile_2x1_inverted_";
#endif

  solid_solver.outputStateToDisk(outputFilename);

  int num_steps = 20;
  double t    = 0.0;
  double tmax = 1.0;
  double dt   = tmax / num_steps;
  // double gblDispYmin;

  for (int i = 0; i < num_steps; i++) {

    t += dt;
    // orderParam = max_order_param * (tmax - t) / tmax;
    orderParam = max_order_param * std::pow((tmax - t) / tmax, 1.0);

    if (rank == 0) {
      std::cout << "\n\n............................"
                << "\n... Entering time step: " << i + 1 << " (/" << num_steps << ")"
                << "\n............................\n"
                << "\n... Using order parameter: " << max_order_param * (tmax - t) / tmax
                << "\n... Using gamma = " << gamma_angle << ", and eta = " << eta_angle << std::endl;
    }

    solid_solver.advanceTimestep(dt);
    solid_solver.outputStateToDisk(outputFilename);

    auto&                 fes             = solid_solver.displacement().space();
    mfem::ParGridFunction displacement_gf = solid_solver.displacement().gridFunction();
    int                   numDofs         = fes.GetNDofs();
    mfem::Vector          dispVecX(numDofs);
    dispVecX = 0.0;
    mfem::Vector dispVecY(numDofs);
    dispVecY = 0.0;
#ifndef USE_2D_MESH
    mfem::Vector dispVecZ(numDofs);
    dispVecZ = 0.0;
#endif

    for (int k = 0; k < numDofs; k++) {
      dispVecX(k) = displacement_gf(0 * numDofs + k);
      dispVecY(k) = displacement_gf(1 * numDofs + k);
#ifndef USE_2D_MESH
      dispVecZ(k) = displacement_gf(2 * numDofs + k);
#endif
    }
    double gblDispXmin, lclDispXmin = dispVecX.Min();
    double gblDispXmax, lclDispXmax = dispVecX.Max();
    double gblDispYmin, lclDispYmin = dispVecY.Min();
    double gblDispYmax, lclDispYmax = dispVecY.Max();
    MPI_Allreduce(&lclDispXmin, &gblDispXmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&lclDispXmax, &gblDispXmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&lclDispYmin, &gblDispYmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&lclDispYmax, &gblDispYmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

#ifndef USE_2D_MESH
    double gblDispZmin, lclDispZmin = dispVecZ.Min();
    double gblDispZmax, lclDispZmax = dispVecZ.Max();
    MPI_Allreduce(&lclDispZmin, &gblDispZmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&lclDispZmax, &gblDispZmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif

    if (rank == 0) {
      std::cout << "\n... In time step: " << i + 1 << " (/" << num_steps << ")"
                << "\n... Min X displacement: " << gblDispXmin << "\n... Max X displacement: " << gblDispXmax
                << "\n... Min Y displacement: " << gblDispYmin << "\n... Max Y displacement: " << gblDispYmax
#ifndef USE_2D_MESH
                << "\n... Min Z displacement: " << gblDispZmin << "\n... Max Z displacement: " << gblDispZmax
#endif
                << std::endl;

      if (std::isnan(gblDispXmax) || gblDispXmax > 1.0e3) {
        std::cout << "... Solution blew up... Check boundary and initial conditions." << std::endl;
        exit(1);
      }
    }
  }

  serac::exitGracefully();
}
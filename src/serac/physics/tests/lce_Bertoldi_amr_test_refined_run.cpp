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

const static int problemID = 2;

int main(int argc, char* argv[])
{
  auto [num_procs, rank] = serac::initialize(argc, argv);

  constexpr int p = 1;
  constexpr int dim      = 3;

  int serial_refinement   = 0;
  int parallel_refinement = 0;

  // ---------------------------
  // ---------------------------
  // Initial run (original mesh)
  // ---------------------------
  // ---------------------------

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_lce_functional");

  // Construct the appropriate dimension mesh and give it to the data store

  std::string   filename = SERAC_REPO_DIR "/build/tests/refinedMeshAMRTest.mesh";
  auto initial_mesh = buildMeshFromFile(filename);
  auto mesh = mesh::refineAndDistribute(std::move(initial_mesh), serial_refinement, parallel_refinement);

  std::string mesh_tag{"mesh}"};
  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  // Construct a functional-based solid mechanics solver
  LinearSolverOptions linear_options = {.linear_solver = LinearSolver::SuperLU};

  // LinearSolverOptions linear_options = {.linear_solver  = LinearSolver::GMRES,
  //                                                     .preconditioner = Preconditioner::HypreAMG,
  //                                                     .relative_tol   = 1.0e-6,
  //                                                     .absolute_tol   = 1.0e-10,
  //                                                     .max_iterations = 500,
  //                                                     .print_level    = 0};

  NonlinearSolverOptions nonlinear_options = {.nonlin_solver  = serac::NonlinearSolver::Newton,
                                              .relative_tol   = 1.0e-8,
                                              .absolute_tol   = 1.0e-14,
                                              .max_iterations = 12,
                                              .print_level    = 1};
  SolidMechanics<p, dim, Parameters<L2<p>, L2<p>, L2<p> > > solid_solver(
      nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options, GeometricNonlinearities::On, "lce_solid_functional", mesh_tag);

  // Material properties
  double density         = 1.0;    // [Kg / mm3]
  double young_modulus   = 4.0e5;  // 4.0e5 [Kg /s2 / mm]
  double possion_ratio   = 0.45;   // 0.49;   // 0.48 // 
  double beta_param      = 2.0e5; // 5.20e5; // 2.31e5; // [Kg /s2 / mm] 
  double max_order_param = 0.45;   // 0.20;   // 0.45; //
  double min_order_param = 0.0;   // 0.20;   // 0.45; //
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
      double d    = 5.0e-3;
      double t    = 0.525e-3;

      // vertical
      if ( (x[0] < 0.3e-3) || (x[0] > 9.20e-3) || ((x[0] > 4.45e-3)&&(x[0] < 5.05e-3)  ) ){
        return M_PI_2;
      }
      // forward incline
      else if ( x[0] <= 4.5e-3 + 0.0*d*t ){
        return -0.1920;
      }
      // backward incline
      else{ // if ( x[0] >= 5.05e-3 + 0.0*d*t ){
        return +0.1920;
      }
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
  LiquidCrystalElastomerBertoldi lceMat(density, young_modulus, possion_ratio, max_order_param, beta_param);

  solid_solver.setMaterial(DependsOn<ORDER_INDEX, GAMMA_INDEX, ETA_INDEX>{}, lceMat);

  auto zeroFunc = [](const mfem::Vector /*x*/) { return 0.0; };
  solid_solver.setDisplacementBCs({1}, zeroFunc, 0);  // left face x-dir disp = 0
  solid_solver.setDisplacementBCs({2}, zeroFunc, 1);  // bottom face y-dir disp = 0
  solid_solver.setDisplacementBCs({3}, zeroFunc, 2);  // back face z-dir disp = 0

  double iniDispVal = 5.0e-6;
  if (problemID == 4) {
    iniDispVal = 5.0e-8;
  }
  auto ini_displacement = [iniDispVal](const mfem::Vector&, mfem::Vector& u) -> void { u = iniDispVal; };
  solid_solver.setDisplacement(ini_displacement);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  std::string outputFilename = "sol_lce_bertoldi_amr_test_refined_mesh";
  solid_solver.outputStateToDisk(outputFilename);

  // QoI for output
  // --------------
  // auto& pmesh = serac::StateManager::mesh();
  Functional<double(H1<p, dim>, serac::L2<p>, serac::L2<p>, serac::L2<p>)> strainEnergyQoI(
      {&solid_solver.displacement().space(), &orderParam.space(), &gammaParam.space(), &etaParam.space()});
  strainEnergyQoI.AddDomainIntegral(
      serac::Dimension<dim>{},
      DependsOn<0, 1, 2, 3>{},
      [=](double /*t*/, auto /*x*/, auto displacement, auto order_param_tuple, auto gamma_param_tuple, auto eta_param_tuple) {
        auto du_dx = serac::get<1>(displacement);
        serac::LiquidCrystalElastomerBertoldi::State state{};
        // auto strain = serac::sym(du_dx);
        // auto stress = lceMat(state, du_dx, order_param_tuple, gamma_param_tuple, eta_param_tuple);
        // return 0.5 * serac::double_dot(strain, stress);
        auto strainEnergy = lceMat.calculateStrainEnergy(state, du_dx, order_param_tuple, gamma_param_tuple, eta_param_tuple);
        return strainEnergy;
      },
      pmesh);

  // Time stepping
  // --------------    
  int num_steps = 10;
  double t    = 0.0;
  double tmax = 1.0;
  double dt   = tmax / num_steps;
  // double gblDispYmin;

  for (int i = 0; i < num_steps; i++) {

    t += dt;
    // orderParam = max_order_param * (tmax - t) / tmax;
    orderParam = min_order_param + (max_order_param-min_order_param) * std::pow((tmax - t) / tmax, 1.0);

    if (rank == 0) {
      std::cout << "\n\n............................"
                << "\n... Entering time step: " << i + 1 << " (/" << num_steps << ")"
                << "\n............................\n"
                << "\n... Using order parameter: " << max_order_param * (tmax - t) / tmax
                << "\n... Using gamma = " << gamma_angle << ", and eta = " << eta_angle << std::endl;
    }

    solid_solver.advanceTimestep(dt);
    solid_solver.outputStateToDisk(outputFilename);

    // Compute QoI
    double current_qoi = strainEnergyQoI(t, solid_solver.displacement(), orderParam, gammaParam, etaParam);

    // Construct adjoint load
    serac::FiniteElementDual adjoint_load(solid_solver.displacement().space(), "adjoint_load");
    auto dqoi_du = get<1>(strainEnergyQoI(DifferentiateWRT<0>{}, t, solid_solver.displacement(), orderParam, gammaParam, etaParam));
    adjoint_load = *assemble(dqoi_du);

    // Solve adjoint problem
    solid_solver.setAdjointLoad({{"displacement", adjoint_load}});
    solid_solver.reverseAdjointTimestep();    

    // Output data
    auto&                 fes             = solid_solver.displacement().space();
    mfem::ParGridFunction displacement_gf = solid_solver.displacement().gridFunction();
    int                   numDofs         = fes.GetNDofs();
    mfem::Vector          dispVecX(numDofs);
    dispVecX = 0.0;
    mfem::Vector dispVecY(numDofs);
    dispVecY = 0.0;
    mfem::Vector dispVecZ(numDofs);
    dispVecZ = 0.0;

    for (int k = 0; k < numDofs; k++) {
      dispVecX(k) = displacement_gf(0 * numDofs + k);
      dispVecY(k) = displacement_gf(1 * numDofs + k);
      dispVecZ(k) = displacement_gf(2 * numDofs + k);
    }
    double gblDispXmin, lclDispXmin = dispVecX.Min();
    double gblDispXmax, lclDispXmax = dispVecX.Max();
    double gblDispYmin, lclDispYmin = dispVecY.Min();
    double gblDispYmax, lclDispYmax = dispVecY.Max();
    MPI_Allreduce(&lclDispXmin, &gblDispXmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&lclDispXmax, &gblDispXmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&lclDispYmin, &gblDispYmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&lclDispYmax, &gblDispYmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    double gblDispZmin, lclDispZmin = dispVecZ.Min();
    double gblDispZmax, lclDispZmax = dispVecZ.Max();
    MPI_Allreduce(&lclDispZmin, &gblDispZmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&lclDispZmax, &gblDispZmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
      std::cout << "\n... In time step: " << i + 1 << " (/" << num_steps << ")"
                << "\n... Min X displacement: " << gblDispXmin << "\n... Max X displacement: " << gblDispXmax
                << "\n... Min Y displacement: " << gblDispYmin << "\n... Max Y displacement: " << gblDispYmax
                << "\n... Min Z displacement: " << gblDispZmin << "\n... Max Z displacement: " << gblDispZmax
                << std::endl;

    std::cout << "\n... The QoIVal is: " << current_qoi << std::endl;
    
      if (std::isnan(gblDispXmax) || gblDispXmax > 1.0e3) {
        std::cout << "... Solution blew up... Check boundary and initial conditions." << std::endl;
        exit(1);
      }
    }
  }

  serac::exitGracefully();
}
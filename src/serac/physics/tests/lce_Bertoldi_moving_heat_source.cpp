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

const static int problemID = 0;

int main(int argc, char* argv[])
{
  auto [num_procs, rank] = serac::initialize(argc, argv);

  constexpr int p = 1;
  constexpr int dim = 3;
  int serial_refinement   = 0;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_lce_functional");

  // Construct the appropriate dimension mesh and give it to the data store
  double lx = 6.0e-3, ly = 0.5e-3, lz = 0.1667e-3;
  int nElem = 4;
  mfem::Mesh cuboid =
      mfem::Mesh(mfem::Mesh::MakeCartesian3D(36*nElem, 3*nElem, nElem, mfem::Element::HEXAHEDRON, lx, ly, lz));

  auto mesh = mesh::refineAndDistribute(std::move(cuboid), serial_refinement, parallel_refinement);

  serac::StateManager::setMesh(std::move(mesh));

  // Construct a functional-based solid mechanics solver
  LinearSolverOptions linear_options = {.linear_solver = LinearSolver::SuperLU};
  NonlinearSolverOptions nonlinear_options = {.nonlin_solver  = serac::NonlinearSolver::Newton,
                                              .relative_tol   = 1.0e-8,
                                              .absolute_tol   = 1.0e-14,
                                              .max_iterations = 6,
                                              .print_level    = 1};

  // .nonlin_solver = serac::NonlinearSolver::LBFGS};
  // .nonlin_solver = serac::NonlinearSolver::KINFullStep};
  //.nonlin_solver = serac::NonlinearSolver::KINBacktrackingLineSearch};
  // .nonlin_solver = serac::NonlinearSolver::KINPicard};
  // .nonlin_solver = serac::NonlinearSolver::KINFP};

  SolidMechanics<p, dim, Parameters<H1<p>, L2<p>, L2<p> > > solid_solver(
      nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options, GeometricNonlinearities::On, "lce_solid_functional");

  // Material properties
  double density         = 1.0;
  double young_modulus   = 0.1e6; 
  double possion_ratio   = 0.48;
  double beta_param      = 4.0e4;
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
  FiniteElementState orderParam(StateManager::newState(FiniteElementState::Options{.order = p, .name = "orderParam"}));
  auto orderFunc = [=](const mfem::Vector& x, double t) -> double {
    // Define the order function of time and space
    using std::sin;
    auto scaledX = x[0]/lx;
    auto scaledT = t;
    return max_order_param * 0.5 * (1.0 + sin(10.0*(scaledX*scaledX + scaledT*scaledT)) );    
    // return max_order_param * 0.5 * (1.0 + sin(5.0*(1.0*scaledX + 4.0*scaledT)) );
    // return max_order_param * 0.5 * (1.0 + sin(8.0*(2.0*scaledX+0.6)*(scaledT+0.4)) );
  };
  mfem::FunctionCoefficient orderCoef(orderFunc);
  orderParam.project(orderCoef);

  // Parameter 2
  FiniteElementState gammaParam(StateManager::newState(
      FiniteElementState::Options{.order = p, .element_type = ElementType::L2, .name = "gammaParam"}));
  bool               heterogeneousGammaField = problemID == 2 ? true : false;
  auto               gammaFunc = [heterogeneousGammaField, gamma_angle](const mfem::Vector& /*x*/, double) -> double {
      return gamma_angle;
  };
  mfem::FunctionCoefficient gammaCoef(gammaFunc);
  gammaParam.project(gammaCoef);

  // Paremetr 3
  FiniteElementState        etaParam(StateManager::newState(
      FiniteElementState::Options{.order = p, .element_type = ElementType::L2, .name = "etaParam"}));
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
  solid_solver.setDisplacementBCs({1}, zeroFunc, 2);  // bottom face y-dir disp = 0
  solid_solver.setDisplacementBCs({2}, zeroFunc, 1);  // left face x-dir disp = 0
  solid_solver.setDisplacementBCs({5}, zeroFunc, 0);  // back face z-dir disp = 0

  double iniDispVal = 5.0e-6;
  auto ini_displacement = [iniDispVal](const mfem::Vector&, mfem::Vector& u) -> void { u = iniDispVal; };
  solid_solver.setDisplacement(ini_displacement);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  int num_steps = 200;

  std::string outputFilename = "sol_lce_bertoldi_moving_heat_source";
  solid_solver.outputState(outputFilename);

  double t    = 0.0;
  double tmax = 1.0;
  double dt   = tmax / num_steps;

  for (int i = 0; i < num_steps; i++) {
    if (rank == 0) {
      std::cout << "\n\n............................"
                << "\n... Entering time step: " << i + 1 << " (/" << num_steps << ")"
                << "\n............................\n"
                << "\n... Using order parameter: " << max_order_param * (tmax - t) / tmax
                << "\n... Using gamma = " << gamma_angle << ", and eta = " << eta_angle << std::endl;
    }

    solid_solver.advanceTimestep(dt);
    solid_solver.outputState(outputFilename);

    auto&                 fes             = solid_solver.displacement().space();
    mfem::ParGridFunction displacement_gf = solid_solver.displacement().gridFunction();
    int                   numDofs         = fes.GetNDofs();
    mfem::Vector dispVecX(numDofs); dispVecX = 0.0;
    mfem::Vector dispVecY(numDofs); dispVecY = 0.0;
    mfem::Vector dispVecZ(numDofs); dispVecZ = 0.0;

    for (int k = 0; k < numDofs; k++) {
      dispVecX(k) = displacement_gf(0 * numDofs + k);
      dispVecY(k) = displacement_gf(1 * numDofs + k);
      dispVecZ(k) = displacement_gf(2 * numDofs + k);
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

    if (rank == 0) {
      std::cout << "\n... In time step: " << i + 1 << " (/" << num_steps << ")"
                << "\n... Min X displacement: " << gblDispXmin << "\n... Max X displacement: " << gblDispXmax
                << "\n... Min Y displacement: " << gblDispYmin << "\n... Max Y displacement: " << gblDispYmax
                << "\n... Min Z displacement: " << gblDispZmin << "\n... Max Z displacement: " << gblDispZmax
                << std::endl;

      if (std::isnan(gblDispXmax) || gblDispXmax > 1.0e3) {
        std::cout << "... Solution blew up... Check boundary and initial conditions." << std::endl;
        exit(1);
      }
    }

    t += dt;
    orderCoef.SetTime(t);
    orderParam.project(orderCoef);
    // orderParam = max_order_param * (tmax - t) / tmax;
  }

  serac::exitGracefully();
}
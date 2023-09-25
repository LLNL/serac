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

#include "serac/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "serac/physics/boundary_conditions/boundary_condition_helper.hpp"

using namespace serac;

// #define ALT_ITER_SOLVER
#undef ALT_ITER_SOLVER 

const static int problemID = 4;

int main(int argc, char* argv[])
{
  auto [num_procs, rank] = serac::initialize(argc, argv);

  constexpr int p         = 1;
  constexpr int dim       = 3;
  int serial_refinement   = 0;
  int parallel_refinement = 0;

  if (problemID<3){parallel_refinement = 0;}

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_lce_functional");

  // Get mesh
  std::string inputFilename;
  switch (problemID) {
    case 0:
      inputFilename = SERAC_REPO_DIR "/data/meshes/reEntrantHoneycomb_newGeometry_noBorders_verticalSym.g";;
      break;
    case 1:
      inputFilename = SERAC_REPO_DIR "/data/meshes/reEntrantHoneycomb_newGeometry_noBorders_horizontalSym.g";
      break;
    case 2:
      inputFilename = SERAC_REPO_DIR "/data/meshes/reEntrantHoneycomb_newGeometry_noBorders_quarterSym.g";
      break;
    case 3:
      inputFilename = SERAC_REPO_DIR "/data/meshes/dbgLogPileHalfSymm.g";
      break;
    case 4:
      inputFilename = SERAC_REPO_DIR "/data/meshes/halfDomainLogPileVertical.g";
      break;
    default:
      std::cout << "...... Wrong problem ID ......" << std::endl;
      exit(0);
  }

  auto initial_mesh = buildMeshFromFile(inputFilename);
  auto mesh = mesh::refineAndDistribute(std::move(initial_mesh), serial_refinement, parallel_refinement);

  auto pmesh = serac::StateManager::setMesh(std::move(mesh));

  // Construct a functional-based solid mechanics solver
  // LinearSolverOptions linear_options = {.linear_solver = LinearSolver::SuperLU};
  const LinearSolverOptions linear_options = {.linear_solver = LinearSolver::Strumpack, .print_level = 0};

  NonlinearSolverOptions nonlinear_options = {.nonlin_solver  = serac::NonlinearSolver::Newton,
                                              .relative_tol   = 1.0e-8,
                                              .absolute_tol   = 1.0e-12,
                                              .max_iterations = 20,
                                              .print_level    = 1};
  SolidMechanics<p, dim, Parameters<H1<p>, L2<p>, L2<p> > > solid_solver(
      nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options, GeometricNonlinearities::On, "lce_solid_functional");

  // Material properties
  double density         = 1.0;    // [Kg / mm3]
  double young_modulus   = 4.0e5;  // 4.0e5 [Kg /s2 / mm]
  double possion_ratio   = 0.49;   // 0.49;   // 0.48 // 
  double beta_param      = 1.0e5; // 5.20e5; // 2.31e5; // [Kg /s2 / mm] 
  double max_order_param = 0.40;   // 0.20;   // 0.45; //
  double gamma_angle     = M_PI_2;
  double eta_angle       = 0.0;

  // Parameter 1
  FiniteElementState orderParam(StateManager::newState(FiniteElementState::Options{.order = p, .name = "orderParam"}));
  orderParam = max_order_param;

  // Parameter 2
  FiniteElementState gammaParam(StateManager::newState(
      FiniteElementState::Options{.order = p, .element_type = ElementType::L2, .name = "gammaParam"}));
  auto gammaFunc = [=](const mfem::Vector& x, double) -> double {
    double alignmentAngle = 0.0;
    double t = 0.25e-3;
    double L = 6.0e-3;
    double l = 4.0e-3 - t;

    switch (problemID) {
      case 0:
      {
        // vertical walls
        if (x[0]<=t || x[0]>=L-2*t      // first and last colums
        || (x[0]>=l/2-t && x[0]<=l/2+t) // second column
        || (x[0]>=l-t && x[0]<=l+t) ) { // third column
          alignmentAngle = M_PI_2;
        }
        // upwards inclined (excluding vertical walls)
        else if ( (x[1]>l             && (x[0]<l/2 || x[0]>l) )
        || ( (x[1]>0.0 && x[1]<l/2)   && (x[0]<l/2 || x[0]>l) )
        || ( (x[1]>-l && x[1]<-l/2)   && (x[0]<l/2 || x[0]>l) )
        || ( x[1]<=-l               && (x[0]>=l/2 && x[0]<=l) ) 
        || ( x[1]>-l/2 && x[1]<=0.0 && (x[0]>=l/2 && x[0]<=l) )
        || ( x[1]>l/2 && x[1]<=l    && (x[0]>=l/2 && x[0]<=l) )
        ) {
          alignmentAngle = 0.1920; // 11.31 degrees
        }
        // downwards incline (excluding vertical walls)
        else {
          alignmentAngle = -0.1920; // -11.31 degrees
        }
        break;
      }

      case 1:
      {
        // vertical walls
        if ( x[0]<=-(L-2*t) || x[0]>=L-2*t // first and last columns
        || (x[0]>=l-t && x[0]<=l+t)        // second column
        || (x[0]>=l/2-t && x[0]<=l/2+t)    // third column
        || (x[0]>=0.0-t && x[0]<=0.0+t)    // fourth column
        || (x[0]>=-l/2-t && x[0]<=-l/2+t)  // fifth column
        || (x[0]>=-l-t && x[0]<=-l+t) ) {  // sixth column
          alignmentAngle = M_PI_2;
        }
        // upwards inclined (excluding vertical walls)
        else if ( (x[0]<-l          && (x[1]>l || x[1]<l/2) )
        || ( (x[0]>-l && x[0]<-l/2) && (x[1]<=l && x[1]>=l/2) )
        || ( x[0]>-l/2 && x[0]<=0.0 && (x[1]>l || x[1]<l/2) )
        || ( (x[0]>0.0 && x[0]<l/2) && (x[1]<=l && x[1]>=l/2)  )
        || ( x[0]>l/2 && x[0]<=l    && (x[1]>l || x[1]<l/2) )
        || ( x[0]>l                 && (x[1]<=l && x[1]>=l/2) ) 
        ) {
          alignmentAngle = -0.1920; // -11.31 degrees
        }
        // downwards incline (excluding vertical walls)
        else {
          alignmentAngle = 0.1920; // 11.31 degrees
        }
        break;
      }

      case 2:
      {
        // vertical walls
        if (x[0]<=t || x[0]>=L-2*t      // first and last colums
        || (x[0]>=l/2-t && x[0]<=l/2+t) // second column
        || (x[0]>=l-t && x[0]<=l+t) ) { // third column
          alignmentAngle = M_PI_2;
        }
        // upwards inclined (excluding vertical walls)
        else if ( ( (x[0]<l/2)      && (x[1]<=l && x[1]>=l/2)  )
        || ( x[0]>l/2 && x[0]<=l    && (x[1]>l || x[1]<l/2) )
        || ( x[0]>l                 && (x[1]<=l && x[1]>=l/2) ) 
        ) {
          alignmentAngle = -0.1920; // -11.31 degrees
        }
        // downwards incline (excluding vertical walls)
        else {
          alignmentAngle = 0.1920; // 11.31 degrees
        }
        break;
      }

      case 3:
      {
        if ((x[0] >= 1.70e-3 && x[0] <= 2.00e-3) && (
          (x[2] <= 0.20e-3) ||
          (x[2] >= 0.30e-3 && x[2] <= 0.30e-3+0.20e-3) ||
          (x[2] >= 0.60e-3 && x[2] <= 0.60e-3+0.20e-3)
        )) { 
          alignmentAngle = M_PI_2;
        }
        else if ( x[0] >= 5.70e-3 ) { 
          alignmentAngle = M_PI_2;
        }
        else
        {
          alignmentAngle = 0.0;
        }
        break;
      }
      
      case 4:
      {
        if ((x[0] >= 1.70e-3 && x[0] <= 2.00e-3) && (
          (x[2] <= 0.20e-3) ||
          (x[2] >= 0.30e-3 && x[2] <= 0.30e-3+0.20e-3) ||
          (x[2] >= 0.60e-3 && x[2] <= 0.60e-3+0.20e-3) ||
          (x[2] >= 0.90e-3 && x[2] <= 0.90e-3+0.20e-3) ||
          (x[2] >= 1.20e-3 && x[2] <= 1.20e-3+0.20e-3) ||
          (x[2] >= 1.50e-3 && x[2] <= 1.50e-3+0.20e-3) ||
          (x[2] >= 1.80e-3 && x[2] <= 1.80e-3+0.20e-3) ||
          (x[2] >= 2.10e-3 && x[2] <= 2.10e-3+0.20e-3) ||
          (x[2] >= 2.40e-3 && x[2] <= 2.40e-3+0.20e-3) ||
          (x[2] >= 2.70e-3 && x[2] <= 2.70e-3+0.20e-3) ||
          (x[2] >= 3.00e-3 && x[2] <= 3.00e-3+0.20e-3) ||
          (x[2] >= 3.30e-3 && x[2] <= 3.30e-3+0.20e-3) ||
          (x[2] >= 3.60e-3 && x[2] <= 3.60e-3+0.20e-3) ||
          (x[2] >= 3.90e-3 && x[2] <= 3.90e-3+0.20e-3) ||
          (x[2] >= 4.20e-3 && x[2] <= 4.20e-3+0.20e-3) ||
          (x[2] >= 4.50e-3 && x[2] <= 4.50e-3+0.20e-3) ||
          (x[2] >= 4.80e-3 && x[2] <= 4.80e-3+0.20e-3) ||
          (x[2] >= 5.10e-3 && x[2] <= 5.10e-3+0.20e-3) ||
          (x[2] >= 5.40e-3 && x[2] <= 5.40e-3+0.20e-3) ||
          (x[2] >= 5.70e-3 && x[2] <= 5.70e-3+0.20e-3) ||
          (x[2] >= 6.00e-3 && x[2] <= 6.00e-3+0.20e-3) ||
          (x[2] >= 6.30e-3 && x[2] <= 6.30e-3+0.20e-3) ||
          (x[2] >= 6.60e-3 && x[2] <= 6.60e-3+0.20e-3) ||
          (x[2] >= 6.90e-3 && x[2] <= 6.90e-3+0.20e-3)
        )) { 
          alignmentAngle = M_PI_2;
        }
        else if ( x[0] >= 5.70e-3 ) { 
          alignmentAngle = M_PI_2;
        }
        else
        {
          alignmentAngle = 0.0;
        }
        break;
      }

      default:
      {
          std::cout << "...... Wrong problem ID ......" << std::endl;
          exit(0);
      }
    }

    return alignmentAngle;
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
  LiquidCrystalElastomerBertoldi lceMat(density, young_modulus, possion_ratio, max_order_param, beta_param);

  solid_solver.setMaterial(DependsOn<ORDER_INDEX, GAMMA_INDEX, ETA_INDEX>{}, lceMat);

  auto zeroFunc = [](const mfem::Vector /*x*/) { return 0.0; };
  solid_solver.setDisplacementBCs({1}, zeroFunc, 0);  // left face x-dir disp = 0
  if (problemID<3)
  {
    solid_solver.setDisplacementBCs({2}, zeroFunc, 1);  // bottom face y-dir disp = 0
    solid_solver.setDisplacementBCs({3}, zeroFunc, 2);  // back face z-dir disp = 0

    // Generate a true dof set from the boundary attribute

    ////////////////////////////////////////////////////
    // debug:
    // auto is_on_top = [](const mfem::Vector& x) {
    //   if (x(1) > 1.0e-3) {
    //     return true;
    //   }
    //   return false;
    // };
    // auto scalar_offset = [](const mfem::Vector&) { return -0.00001; };
    // solid_solver.setDisplacementBCs(is_on_top, scalar_offset, 2);
    ////////////////////////////////////////////////////

    int attribute = 4;
    mfem::Array<int> elem_attr_is_ess(pmesh->attributes.Max());
    elem_attr_is_ess                = 0;
    elem_attr_is_ess[attribute - 1] = 1;
    mfem::Array<int> ess_tdof_list;

    mfem::H1_FECollection       h1_fec(1, dim);
    mfem::ParFiniteElementSpace h1_fes(pmesh, &h1_fec, 1);
    // serac::mfem_ext::GetEssentialTrueDofsFromElementAttribute(h1_fes, elem_attr_is_ess, ess_tdof_list, 1);
    h1_fes.GetEssentialTrueDofs(elem_attr_is_ess, ess_tdof_list, 1);

    double maxYDisp = 1.0e-3;
    auto is_on_top = [=](const mfem::Vector&, double t, mfem::Vector&) {
      return maxYDisp*(1.0+t);
    };

    solid_solver.setDisplacementBCsByDofList(ess_tdof_list, is_on_top);
  }
  else
  {
    auto is_on_bottom = [=](const mfem::Vector& x) {
      bool tag = false;
      if (x(1) < -5.9e-3) {
        tag = true;
      }
      return tag;
    };

    auto is_at_center_bottom = [=](const mfem::Vector& x) {
      bool tag = false;
      if (problemID==3)
      {
        if (x(1) < -5.9e-3 && x(2) > 6.2e-4 && x(2) < 6.3e-4) {
          tag = true;
        }
      }
      else
      {
        if (x(1) < -5.9e-3 && x(2) > 3.6e-5 && x(2) < 3.7e-5) {
          tag = true;
        }
      }
      return tag;
    };
  
    auto zero_scalar = [](const mfem::Vector&) { return 0.0; };
    solid_solver.setDisplacementBCs(is_on_bottom, zero_scalar, 1);
    solid_solver.setDisplacementBCs(is_at_center_bottom, zero_scalar, 2);

    auto is_on_top = [=](const mfem::Vector& x) {
      bool tag = false;
      if (x(1) > 5.9e-3) {
        tag = true;
      }
      return tag;
    };

    auto scalar_offset = [](const mfem::Vector&, double t) { return -0.0015*t; };
    solid_solver.setDisplacementBCs(is_on_top, scalar_offset, 1);

  }

  // solid_solver.setDisplacementBCs({3}, zeroFunc, 2);  // back face z-dir disp = 0
  // solid_solver.setDisplacementBCs({6}, zeroFunc, 2);  // back face z-dir disp = 0
  
  // auto nonZeroFunc = [](const mfem::Vector /*x*/) { return -3.6e-3; };
  // solid_solver.setDisplacementBCs({4}, nonZeroFunc, 1);  // back face z-dir disp = 0

  double iniDispVal = 1.0e-8;
  auto ini_displacement = [=](const mfem::Vector&, mfem::Vector& u) -> void { u = iniDispVal; };
  solid_solver.setDisplacement(ini_displacement);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  std::string outputFilename;
  switch (problemID) {
    case 0:
      outputFilename = "sol_honeycomb_3x3_compression_vertical";
      break;
    case 1:
      outputFilename = "sol_honeycomb_3x3_compression_horizontal";
      break;
    case 2:
      outputFilename = "sol_honeycomb_3x3_compression_quarter";
      break;
    case 3:
      outputFilename = "sol_logpile_3x3_compression_dbg";
      break;
    case 4:
      outputFilename = "sol_logpile_3x3_compression";
      break;
    default:
      std::cout << "...... Wrong problem ID ......" << std::endl;
      exit(0);
  }

  solid_solver.outputState(outputFilename);

  int num_steps = 50;
  double t    = 0.0;
  double tmax = 1.0;
  double dt   = tmax / num_steps;

  for (int i = 0; i < num_steps; i++) {
    // orderParam = max_order_param * (tmax - t) / tmax;
    // orderParam = max_order_param * std::pow((tmax - t) / tmax, 1.0);

    if (rank == 0) {
      std::cout << "\n\n............................"
                << "\n... Entering time step: " << i + 1 << " (/" << num_steps << ")"
                << "\n............................\n"
                << "\n... Using order parameter: " << max_order_param << ", gamma = " << gamma_angle << ", and eta = " << eta_angle
                // << "\n... Using displacement = " << maxYDisp * t / tmax 
                << std::endl;
    }

    solid_solver.advanceTimestep(dt);
    solid_solver.outputState(outputFilename);

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
  }

  serac::exitGracefully();
}
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

int main(int argc, char* argv[])
{
  // MPI_Init(&argc, &argv);
  // int rank = -1;
  // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  auto [num_procs, rank] = serac::initialize(argc, argv);

  // axom::slic::SimpleLogger logger;
  axom::slic::setIsRoot(rank == 0);

  constexpr int p                   = 1;
  constexpr int dim                 = 3;
  int           serial_refinement   = 0;
  int           parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "LCE_free_swelling_bertoldi");

  // Construct the appropriate dimension mesh and give it to the data store
  int nElem = 8;
  // double lx = 2.5e-3, ly = 0.25e-3, lz = 12.5e-3;
  // ::mfem::Mesh cuboid = mfem::Mesh(mfem::Mesh::MakeCartesian3D(5*nElem, nElem, 25*nElem, mfem::Element::HEXAHEDRON,
  // lx, ly, lz)); double ly = 2.5, lz = 0.25, lx = 12.5;
  // double   lx = (10.0e-3)/2, ly = (5.0e-3)/2, lz = (0.425e-3)/2;
  // log pile material strips: 30 mm x 3 mm x 0.45 mm 
  double   lx = (30.0e-3)/2, ly = (3.0e-3)/2, lz = (0.45e-3)/2;
  ::mfem::Mesh cuboid =
      // mfem::Mesh(mfem::Mesh::MakeCartesian3D(20 * nElem, 10 * nElem, nElem, mfem::Element::HEXAHEDRON, lx, ly, lz));
      mfem::Mesh(mfem::Mesh::MakeCartesian3D(70 * nElem, 7 * nElem, nElem, mfem::Element::HEXAHEDRON, lx, ly, lz));
  auto mesh = mesh::refineAndDistribute(std::move(cuboid), serial_refinement, parallel_refinement);
  std::string mesh_tag{"mesh}"}; 
  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  // Side set ordering for MFEM Cartesian mesh:
  // SS-1: XY bottom plane
  // SS-2: XZ left plane
  // SS-3: YZ front plane
  // SS-4: XZ right plane
  // SS-5: YZ back plane
  // SS-6: XY top plane

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
  // IterativeSolverOptions default_linear_options    = {.rel_tol     = 1.0e-6,
  //                                                  .abs_tol     = 1.0e-16,
  //                                                  .print_level = 0,
  //                                                  .max_iter    = 600,
  //                                                  .lin_solver  = LinearSolver::GMRES,
  //                                                  .prec        = HypreBoomerAMGPrec{}};
  // NonlinearSolverOptions default_nonlinear_options = {
  //     .rel_tol = 1.0e-6, .abs_tol = 1.0e-13, .max_iter = 10, .print_level = 1};

  // LinearSolverOptions linear_options = {.linear_solver = LinearSolver::SuperLU};
  const LinearSolverOptions linear_options = {.linear_solver = LinearSolver::Strumpack, .print_level = 0};

  // LinearSolverOptions linear_options = {.linear_solver  = LinearSolver::GMRES,
  //                                                     .preconditioner = Preconditioner::HypreAMG,
  //                                                     .relative_tol   = 1.0e-6,
  //                                                     .absolute_tol   = 1.0e-10,
  //                                                     .max_iterations = 500,
  //                                                     .print_level    = 0};

  NonlinearSolverOptions nonlinear_options = {.nonlin_solver  = serac::NonlinearSolver::Newton,
                                              .relative_tol   = 1.0e-8,
                                              .absolute_tol   = 1.0e-14,
                                              .max_iterations = 10,
                                              .print_level    = 1};
  SolidMechanics<p, dim, Parameters<H1<p>, L2<p>, L2<p> > > solid_solver(
      nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options, GeometricNonlinearities::On, "lce_solid_free_swelling", mesh_tag);

  // -------------------
  // Material properties
  // -------------------

  double density         = 1.0;    // [Kg / mm3]
  double young_modulus   = 9.34e5;  // 3.0e2;  // [Kg /s2 / mm]
  double possion_ratio   = 0.45;
  double beta_param      = 5.75e5; // 2.9e5; // 5.0e5; // 5.2e5;  // 2.31e5; // [Kg /s2 / mm] 0.041 //
  double max_order_param = 0.40; // 0.2; // 0.45
  double min_order_param = 0.05;
  // -------------------

  // Set material
  LiquidCrystalElastomerBertoldi lceMat(density, young_modulus, possion_ratio, max_order_param, beta_param);

  // Parameter 1
  FiniteElementState orderParam(pmesh, L2<0>{}, "orderParam");
  orderParam = max_order_param;

  // Parameter 2
  FiniteElementState gammaParam(
      pmesh, L2<0>{}, "gammaParam");

  int  lceArrangementTag = 1;
  auto gammaFunc         = [lceArrangementTag](const mfem::Vector& x, double) -> double {
    if (lceArrangementTag == 1) {
      return 0.0;  // M_PI_2;
    } else if (lceArrangementTag == 2) {
      // Gyroid
      double a   = 4;
      double LSF = sin(2 * M_PI / a * x[0]) * cos(2 * M_PI / a * x[1]) +
                   sin(2 * M_PI / a * x[1]) * cos(2 * M_PI / a * x[2]) +
                   sin(2 * M_PI / a * x[2]) * cos(2 * M_PI / a * x[0]);

      return (LSF > 0.0) ? 0.667 * M_PI_2 : 0.333 * M_PI_2;
    } else if (lceArrangementTag == 3) {
      // Straight rods
      double rad       = 0.5;
      double LSF_rod_1 = std::pow(x[0] - 4.0, 2) + std::pow(x[1] - 4.0, 2) - std::pow(rad, 2);
      double LSF_rod_2 = std::pow(x[2] - 4.0, 2) + std::pow(x[1] - 4.0, 2) - std::pow(rad, 2);
      double LSF_rod_3 = std::pow(x[0] - 4.0, 2) + std::pow(x[2] - 4.0, 2) - std::pow(rad, 2);

      // Inclined rod
      // double rotAngle =  M_PI_2/2.0; // 0.785398; // 0.6; //

      // double xp = x[0]; //  x[0]*cos(rotAngle) - x[1]*sin(rotAngle);
      // double yp = x[1]*cos(-rotAngle) - x[2]*sin(-rotAngle); // x[0]*sin(rotAngle) + x[1]*cos(rotAngle);
      // double zp = x[1]*sin(-rotAngle) + x[2]*cos(-rotAngle); // x[2];

      // double xpp =  xp*cos(rotAngle) - yp*sin(rotAngle); // xp;
      // // double ypp = xp*sin(rotAngle) + yp*cos(rotAngle); // yp*cos(-rotAngle) - zp*sin(-rotAngle);
      // double zpp = zp; // yp*sin(-rotAngle) + zp*cos(-rotAngle);

      // double LSF_rod_4 = std::pow(xpp, 2) + std::pow(zpp, 2) - std::pow(rad, 2);

      double LSF_rod_4 =
          std::pow(x[0] - x[1], 2) + std::pow(x[1] - x[2], 2) + std::pow(x[2] - x[0], 2) - 3 * std::pow(rad, 2);

      // Sphere
      double LSF_sph = std::pow(x[0], 2) + std::pow(x[1], 2) + std::pow(x[2], 2) - std::pow(2.75 * rad, 2);

      // Combine LSFs4d
      double final_LSF = std::min(std::min(std::min(std::min(LSF_rod_1, LSF_rod_2), LSF_rod_3), LSF_rod_4), LSF_sph);

      return (final_LSF > 0.0) ? 1.0 * M_PI_2 : 0.0 * M_PI_2;
    } else {
      // Spheres (not ready yet)
      double rad = 0.65;
      return (std::pow(x[0] - 3.0, 2) + std::pow(x[1] - 3.0, 2) - std::pow(rad, 2) < 0.0 ||
              std::pow(x[0] - 1.0, 2) + std::pow(x[1] - 3.0, 2) - std::pow(rad, 2) < 0.0 ||
              std::pow(x[0] - 3.0, 2) + std::pow(x[1] - 1.0, 2) - std::pow(rad, 2) < 0.0 ||
              std::pow(x[0] - 1.0, 2) + std::pow(x[1] - 1.0, 2) - std::pow(rad, 2) < 0.0)
                 ? 0.333 * M_PI_2
                 : 0.667 * M_PI_2;
    }
  };

  mfem::FunctionCoefficient gammaCoef(gammaFunc);
  gammaParam.project(gammaCoef);

  // Paremetr 3
  FiniteElementState etaParam(
      pmesh, L2<0>{}, "etaParam");
  auto                      etaFunc = [](const mfem::Vector& /*x*/, double) -> double { return 0.0; };
  mfem::FunctionCoefficient etaCoef(etaFunc);
  etaParam.project(etaCoef);

  // Set parameters
  constexpr int ORDER_INDEX = 0;
  constexpr int GAMMA_INDEX = 1;
  constexpr int ETA_INDEX   = 2;

  solid_solver.setParameter(ORDER_INDEX, orderParam);
  solid_solver.setParameter(GAMMA_INDEX, gammaParam);
  solid_solver.setParameter(ETA_INDEX, etaParam);

  solid_solver.setMaterial(DependsOn<ORDER_INDEX, GAMMA_INDEX, ETA_INDEX>{}, lceMat);

  // Boundary conditions:
  // Prescribe zero displacement at the supported end of the beam
  auto zero_displacement = [](const mfem::Vector& /*x*/) { return 0.0; };
  solid_solver.setDisplacementBCs({1}, zero_displacement, 2);  // bottom face y-dir disp = 0
  solid_solver.setDisplacementBCs({2}, zero_displacement, 1);  // left face x-dir disp = 0
  solid_solver.setDisplacementBCs({5}, zero_displacement, 0);  // back face z-dir disp = 0

  auto ini_displacement = [](const mfem::Vector& x, mfem::Vector& u) -> void {
    //  u = 0.0000000001;
    u[0] = 0.0001 * x[0];
    u[1] = 0.0001 * x[1];
    u[2] = 0.0001 * x[2];
  };
  solid_solver.setDisplacement(ini_displacement);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  int num_steps = 15;
  double t    = 0.0;
  double tmax = 1.0;
  double dt   = tmax / num_steps;

  solid_solver.advanceTimestep(dt);
  std::string outputFilename = "sol_lce_bertoldi_free_swelling_log_pile_mat";
  solid_solver.outputStateToDisk(outputFilename);


  for (int i = 0; i < num_steps; i++) {
    
    t += dt;
    
    if (rank == 0) {
      std::cout << "\n\n............................"
                << "\n... Entering time step: " << i + 1 << " (/" << num_steps << ")"
                << "\n............................\n"
                << "\n... Using order parameter: " << min_order_param + (max_order_param - min_order_param) * (tmax - t) / tmax
                << "\n... Using two gamma angles" << std::endl;
    }
    solid_solver.advanceTimestep(dt);
    solid_solver.outputStateToDisk(outputFilename);

    // FiniteElementState& displacement = solid_solver.displacement();
    mfem::ParGridFunction displacement_gf = solid_solver.displacement().gridFunction();
    auto&               fes          = solid_solver.displacement().space();
    int                 numDofs      = fes.GetNDofs();
    mfem::Vector dispVecX(numDofs);
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
      std::cout << "\n... Entering time step: " << i + 1 << "\n... At time: " << t
                << "\n... Min X displacement: " << gblDispXmin
                << "\n... Max X displacement: " << gblDispXmax
                << "\n... Min Y displacement: " << gblDispYmin
                << "\n... Max Y displacement: " << gblDispYmax
                << "\n... Min Z displacement: " << gblDispZmin
                << "\n... Max Z displacement: " << gblDispZmax << std::endl;
    }

    if (std::isnan(dispVecX.Max()) || std::isnan(-1 * dispVecX.Max())) {
      if (rank == 0) {
        std::cout << "... Solution blew up... Check boundary and initial conditions." << std::endl;
      }
      exit(1);
    }

    orderParam = min_order_param + (max_order_param - min_order_param) * (tmax - t) / tmax;
  }

  serac::exitGracefully();
}

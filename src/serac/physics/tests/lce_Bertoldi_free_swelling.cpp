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
  int           serial_refinement   = 0;
  int           parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "LCE_free_swelling_bertoldi");

  // Construct the appropriate dimension mesh and give it to the data store
  int nElem = 6;
  // double lx = 2.5e-3, ly = 0.25e-3, lz = 12.5e-3;
  // ::mfem::Mesh cuboid = mfem::Mesh(mfem::Mesh::MakeCartesian3D(5*nElem, nElem, 25*nElem, mfem::Element::HEXAHEDRON,
  // lx, ly, lz)); double ly = 2.5, lz = 0.25, lx = 12.5;
  double       lx = 12.5e-3, ly = 2.5e-3, lz = 0.25e-3;
  ::mfem::Mesh cuboid =
      mfem::Mesh(mfem::Mesh::MakeCartesian3D(25 * nElem, 5 * nElem, nElem, mfem::Element::HEXAHEDRON, lx, ly, lz));
  auto mesh = mesh::refineAndDistribute(std::move(cuboid), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

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
  IterativeSolverOptions default_linear_options    = {.rel_tol     = 1.0e-6,
                                                   .abs_tol     = 1.0e-16,
                                                   .print_level = 0,
                                                   .max_iter    = 600,
                                                   .lin_solver  = LinearSolver::GMRES,
                                                   .prec        = HypreBoomerAMGPrec{}};
  NonlinearSolverOptions default_nonlinear_options = {
      .rel_tol = 1.0e-6, .abs_tol = 1.0e-13, .max_iter = 10, .print_level = 1};
  SolidMechanics<p, dim, Parameters<H1<p>, L2<p>, L2<p> > > solid_solver(
      {default_linear_options, default_nonlinear_options}, GeometricNonlinearities::Off, "lce_solid_functional");
  // SolidMechanics<p, dim, Parameters<H1<p>, L2<p>, L2<p>>> solid_solver(default_static_options,
  // GeometricNonlinearities::Off,
  //                                                                 "lce_solid_functional");

  // -------------------
  // Material properties
  // -------------------

  double density         = 1.0;    // [Kg / mm3]
  double young_modulus   = 4.0e5;  // 3.0e2;  // [Kg /s2 / mm]
  double possion_ratio   = 0.48;
  double beta_param      = 2.0e5;  // 5.1e-2; // 100.0; // [Kg /s2 / mm] 0.041 //
  double max_order_param = 0.2;
  // -------------------

  // Set material
  LiqCrystElast_Bertoldi        lceMat(density, young_modulus, possion_ratio, max_order_param, beta_param);
  LiqCrystElast_Bertoldi::State initial_state{};

  // Parameter 1
  FiniteElementState orderParam(StateManager::newState(FiniteElementState::Options{.order = p, .name = "orderParam"}));
  orderParam = max_order_param;

  // Parameter 2
  FiniteElementState gammaParam(
      StateManager::newState(FiniteElementState::Options{.order = p, .vector_dim = 1, .name = "gammaParam"}));

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
      StateManager::newState(FiniteElementState::Options{.order = p, .vector_dim = 1, .name = "etaParam"}));
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

  auto param_data = solid_solver.createQuadratureDataBuffer(initial_state);
  solid_solver.setMaterial(DependsOn<ORDER_INDEX, GAMMA_INDEX, ETA_INDEX>{}, lceMat, param_data);

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
  int num_steps = 50;

  std::string outputFilename = "sol_lce_bertoldi_free_swelling";
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
                << "\n... Using two gamma angles" << std::endl;
    }

    t += dt;
    solid_solver.advanceTimestep(dt);
    solid_solver.outputState(outputFilename);

    FiniteElementState& displacement = solid_solver.displacement();
    auto&               fes          = displacement.space();

    mfem::Vector dispVecX(fes.GetNDofs());
    dispVecX = 0.0;
    mfem::Vector dispVecY(fes.GetNDofs());
    dispVecY = 0.0;
    mfem::Vector dispVecZ(fes.GetNDofs());
    dispVecZ = 0.0;

    for (int k = 0; k < fes.GetNDofs(); k++) {
      dispVecX(k) = displacement(3 * k + 0);
      dispVecY(k) = displacement(3 * k + 1);
      dispVecZ(k) = displacement(3 * k + 2);
    }

    if (rank == 0) {
      std::cout << "\n... Entering time step: " << i + 1 << "\n... At time: " << t
                << "\n... Min X displacement: " << dispVecX.Min() << "\n... Max X displacement: "
                << dispVecX.Max()
                // <<"\n... Min Y displacement: " << dispVecY.Min()
                << "\n... Max Y displacement: "
                << dispVecY.Max()
                // <<"\n... Min Z displacement: " << dispVecZ.Min()
                << "\n... Max Z displacement: " << dispVecZ.Max() << std::endl;
    }

    if (std::isnan(dispVecX.Max()) || std::isnan(-1 * dispVecX.Max())) {
      if (rank == 0) {
        std::cout << "... Solution blew up... Check boundary and initial conditions." << std::endl;
      }
      exit(1);
    }

    orderParam = max_order_param * (tmax - t) / tmax;
  }

  MPI_Finalize();
}

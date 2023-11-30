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
  int           serial_refinement   = 1;
  int           parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_lce_functional");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex-flat.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
   std::string mesh_tag{"mesh"}; auto& pmesh = serac::StateManager::setMesh(std::move(mesh));

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
  SolidMechanics<p, dim, Parameters<H1<p>, L2<p>, L2<p>>> solid_solver(
      default_static_options, GeometricNonlinearities::Off, "lce_solid_functional");

  // Material properties
  double density         = 1.0;
  double young_modulus   = 1.0;
  double possion_ratio   = 0.49;
  double beta_param      = 0.041;
  double max_order_param = 0.7;

  // Parameter 1
  FiniteElementState orderParam(pmesh, L2<0>{}, "orderParam");
  orderParam = max_order_param;

  // Parameter 2
  FiniteElementState gammaParam(
      StateManager::newState(FiniteElementState::Options{.order = p, .vector_dim = 3, .name = "gammaParam"}));
  auto gammaFunc = [](const mfem::Vector& x, double) -> double { return (x[1] > 0.5) ? M_PI_2 : 0.0; };
  mfem::FunctionCoefficient gammaCoef(gammaFunc);
  gammaParam.project(gammaCoef);

  // Paremetr 3
  FiniteElementState etaParam(
      StateManager::newState(FiniteElementState::Options{.order = p, .vector_dim = 3, .name = "etaParam"}));
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

  // Set material
  LiqCrystElast_Bertoldi        lceMat(density, young_modulus, possion_ratio, max_order_param, beta_param);
  LiqCrystElast_Bertoldi::State initial_state{};

  auto param_data = solid_solver.createQuadratureDataBuffer(initial_state);
  solid_solver.setMaterial(DependsOn<ORDER_INDEX, GAMMA_INDEX, ETA_INDEX>{}, lceMat, param_data);

  // Boundary conditions:
  // Prescribe zero displacement at the supported end of the beam
  std::set<int> support           = {1};
  auto          zero_displacement = [](const mfem::Vector&, mfem::Vector& u) -> void { u = 0.0; };
  solid_solver.setDisplacementBCs(support, zero_displacement);

  solid_solver.setDisplacement(zero_displacement);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  int num_steps = 10;

  solid_solver.outputStateToDisk("sol_lce_bertoldi");

  double t    = 0.0;
  double tmax = 1.0;
  double dt   = tmax / num_steps;
  for (int i = 0; i < num_steps; i++) {
    if (rank == 0) {
      std::cout << "\n\n............................"
                << "\n... Entering time step: " << i + 1 << "\n............................\n"
                << "\n... Using order parameter: " << max_order_param * (tmax - t) / tmax
                << "\n... Using two gamma angles" << std::endl;
    }
    // [[maybe_unused]] auto [K, K_e] = solid_solver.stiffnessMatrix();
    // auto [K, K_e] = solid_solver.stiffnessMatrix();
    // std::cout<<K<<std::endl;
    // std::cout<<K_e<<std::endl;

    mfem::ParGridFunction displacement_gf = solid_solver.displacement().gridFunction();
    if (std::isnan(displacement_gf.Max()) || std::isnan(displacement_gf.Min())) {
      if (rank == 0) {
        std::cout
            << "... Solution blew up... Check boundary and initial conditions. Or maybe bug when running in parallel."
            << std::endl;
      }
      exit(1);
    }

    t += dt;
    solid_solver.advanceTimestep(dt);
    solid_solver.outputStateToDisk("sol_lce_bertoldi");

    orderParam = max_order_param * (tmax - t) / tmax;
  }

  MPI_Finalize();
}

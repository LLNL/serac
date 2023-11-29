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

using namespace serac;

using serac::solid_mechanics::default_static_options;

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  axom::slic::SimpleLogger logger;

  constexpr int p   = 1;
  constexpr int dim = 3;

  int num_steps = 10;

  // int serial_refinement   = 0;
  // int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
#ifdef LOAD_DRIVEN
  serac::StateManager::initialize(datastore, "lce_compression_test_load");
#else
  serac::StateManager::initialize(datastore, "lce_compression_test_temp");
#endif

  // Construct the appropriate dimension mesh and give it to the data store
  // std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex-flat.mesh";
  // std::string filename = SERAC_REPO_DIR "/data/meshes/LCE_tensileTestSpecimen_nonDim.g";
  // auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);

  int          nElem  = 6;
  ::mfem::Mesh cuboid = mfem::Mesh(
      mfem::Mesh::MakeCartesian3D(4 * nElem, 4 * nElem, nElem, mfem::Element::HEXAHEDRON, 8.5 / 2, 8.5 / 2, 2.162 / 2));
  auto mesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, cuboid);
   std::string mesh_tag{"mesh}"}; auto& pmesh = serac::StateManager::setMesh(std::move(mesh));

  double             initial_temperature = 290;  // 300.0;
  double             final_temperature   = 400;  // 430.0;
  FiniteElementState temperature(
      StateManager::newState(FiniteElementState::Options{.order = p, .name = "temperature"}));

  temperature = initial_temperature + 0.0 * final_temperature;

  FiniteElementState gamma(
      StateManager::newState(FiniteElementState::Options{.order = p, .vector_dim = 3, .name = "gamma"}));

  // orient fibers in the beam like below (horizontal when y < 0.5, vertical when y > 0.5):
  //
  // y
  // ^                         4
  // |                         |
  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━┓-- 4
  // ┃ | | | | | | | | | | | | ┃
  // ┃ | | | | | | | | | | | | ┃
  // ┃ | | | | | | | | | | | | ┃
  // ┃ | | | | | | | | | | | | ┃
  // ┃ | | | | | | | | | | | | ┃
  // ┃ - - - - - - - - - - - - ┃
  // ┃ - - - - - - - - - - - - ┃
  // ┃ - - - - - - - - - - - - ┃
  // ┃ - - - - - - - - - - - - ┃
  // ┃ - - - - - - - - - - - - ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━┛--> x

  int  lceArrangementTag = 1;
  auto gamma_func        = [lceArrangementTag](const mfem::Vector& x, double) -> double {
    if (lceArrangementTag == 1) {
      // return (x[0] > 1.0) ? M_PI_2 : 0.0;
      return M_PI_2;
    } else if (lceArrangementTag == 2) {
      return (x[1] > 2.0) ? M_PI_2 : 0.0;
    } else if (lceArrangementTag == 3) {
      return ((x[0] - 2.0) * (x[1] - 2.0) > 0.0) ? 0.333 * M_PI_2 : 0.667 * M_PI_2;
    } else {
      double rad = 0.65;
      return (std::pow(x[0] - 3.0, 2) + std::pow(x[1] - 3.0, 2) - std::pow(rad, 2) < 0.0 ||
              std::pow(x[0] - 1.0, 2) + std::pow(x[1] - 3.0, 2) - std::pow(rad, 2) < 0.0 ||
              std::pow(x[0] - 3.0, 2) + std::pow(x[1] - 1.0, 2) - std::pow(rad, 2) < 0.0 ||
              std::pow(x[0] - 1.0, 2) + std::pow(x[1] - 1.0, 2) - std::pow(rad, 2) < 0.0)
                 ? 0.333 * M_PI_2
                 : 0.667 * M_PI_2;
    }
  };

  mfem::FunctionCoefficient coef(gamma_func);
  gamma.project(coef);

  IterativeSolverOptions default_linear_options    = {.rel_tol     = 1.0e-6,
                                                   .abs_tol     = 1.0e-16,
                                                   .print_level = 0,
                                                   .max_iter    = 500,
                                                   .lin_solver  = LinearSolver::GMRES,
                                                   .prec        = HypreBoomerAMGPrec{}};
  NonlinearSolverOptions default_nonlinear_options = {
      .rel_tol = 1.0e-4, .abs_tol = 1.0e-7, .max_iter = 5, .print_level = 1};

  // Construct a functional-based solid mechanics solver
  // SolidMechanics<p, dim, Parameters< H1<p>, L2<p> > > solid_solver(solid_mechanics::default_static_options,
  // GeometricNonlinearities::Off,
  //                                      "lce_solid_functional");
  SolidMechanics<p, dim, Parameters<H1<p>, L2<p> > > solid_solver({default_linear_options, default_nonlinear_options},
                                                                  GeometricNonlinearities::Off, "lce_solid_functional");

  constexpr int TEMPERATURE_INDEX = 0;
  constexpr int GAMMA_INDEX       = 1;

  solid_solver.setParameter(TEMPERATURE_INDEX, temperature);
  solid_solver.setParameter(GAMMA_INDEX, gamma);

  double density                = 1.0;
  double E                      = 1.0e-1;  // 1.0;
  double nu                     = 0.38;    // 0.49;
  double shear_modulus          = 0.5 * E / (1.0 + nu);
  double bulk_modulus           = E / 3.0 / (1.0 - 2.0 * nu);
  double order_constant         = 10;    // 6.0;
  double order_parameter        = 0.95;  // 0.7;
  double transition_temperature = 348;   // 370.0;
  double Nb2                    = 1.0;

  LiqCrystElast_Brighenti mat(density, shear_modulus, bulk_modulus, order_constant, order_parameter,
                              transition_temperature, Nb2);

  LiqCrystElast_Brighenti::State initial_state{};

  auto qdata = solid_solver.createQuadratureDataBuffer(initial_state);

  solid_solver.setMaterial(DependsOn<TEMPERATURE_INDEX, GAMMA_INDEX>{}, mat, qdata);

  // prescribe symmetry conditions
  auto zeroFunc = [](const mfem::Vector /*x*/) { return 0.0; };
  solid_solver.setDisplacementBCs({1}, zeroFunc, 2);  // bottom face z-dir disp = 0
  solid_solver.setDisplacementBCs({2}, zeroFunc, 1);  // left face x-dir disp = 0
  solid_solver.setDisplacementBCs({5}, zeroFunc, 0);  // back face z-dir disp = 0

#ifdef LOAD_DRIVEN
  auto ini_displacement = [](const mfem::Vector&, mfem::Vector& u) -> void { u = 0.000001; };
#else
  auto        ini_displacement = [](const mfem::Vector&, mfem::Vector& u) -> void { u = 0.001; };
#endif
  solid_solver.setDisplacement(ini_displacement);

  double iniLoadVal = -5.0e-5;
  double loadVal    = iniLoadVal;
  solid_solver.setPiolaTraction([&loadVal](auto x, auto /*n*/, auto /*t*/) {
    return tensor<double, 3>{0, 0, loadVal * (x[2] > (2.16 / 2))};
  });

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
#ifdef LOAD_DRIVEN
  std::string output_filename = "sol_lce_compression_load";
#else
  std::string output_filename  = "sol_lce_compression_temp";
#endif
  solid_solver.outputStateToDisk(output_filename);

  double t    = 0.0;
  double tmax = 1.0;
  double dt   = tmax / num_steps;
  for (int i = 0; i < num_steps; i++) {
    if (rank == 0) {
      std::cout << "\n... Entering time step: " << i + 1 << "\n... At time: " << t
#ifdef LOAD_DRIVEN
                << "\n... And with a compression load of: " << loadVal
#else
                << "\n... And with uniform temperature of: "
                << initial_temperature * (1.0 - (t / tmax)) + final_temperature * (t / tmax)
#endif
                << std::endl;
    }

    solid_solver.advanceTimestep(dt);
    solid_solver.outputStateToDisk(output_filename);

    t += dt;

#ifdef LOAD_DRIVEN
    loadVal = iniLoadVal * 500 * t / tmax;
#else
    temperature = initial_temperature * (1.0 - (t / tmax)) + final_temperature * (t / tmax);
#endif
  }

  MPI_Finalize();
}

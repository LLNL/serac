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
#include "serac/physics/solid_functional.hpp"
#include "serac/physics/materials/liquid_crystal_elastomer.hpp"

using namespace serac;

using serac::solid_mechanics::default_static_options;

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  constexpr int p                   = 1;
  constexpr int dim                 = 3;
  int           serial_refinement   = 2;
  int           parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_functional_static_solve_J2");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex-flat.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  double             initial_temperature = 300.0;
  double             final_temperature   = 400.0;
  FiniteElementState temperature(
      StateManager::newState(FiniteElementState::Options{.order = p, .name = "temperature"}));

  temperature = initial_temperature;

  auto fec = std::unique_ptr<mfem::FiniteElementCollection>(new mfem::L2_FECollection(p, dim));

  FiniteElementState gamma(
      StateManager::newState(FiniteElementState::Options{.order = p, .coll = std::move(fec), .name = "gamma"}));

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
  auto gamma_func = [](const mfem::Vector& x, double) -> double { return (x[1] > 0.5) ? M_PI_2 : 0.0; };

  mfem::FunctionCoefficient coef(gamma_func);
  gamma.project(coef);

  // Construct a functional-based solid mechanics solver
  SolidFunctional<p, dim, Parameters<H1<p>, L2<p> > > solid_solver(default_static_options, GeometricNonlinearities::Off,
                                                                   FinalMeshOption::Reference, "solid_functional",
                                                                   {temperature, gamma});

  double density                = 1.0;
  double E                      = 1.0;
  double nu                     = 0.48;
  double shear_modulus          = 0.5 * E / (1.0 + nu);
  double bulk_modulus           = E / 3.0 / (1.0 - 2.0 * nu);
  double order_constant         = 10.0;
  double order_parameter        = 1.0;
  double transition_temperature = 370.0;
  double Nb2                    = 1.0;

  LiquidCrystalElastomer mat(density, shear_modulus, bulk_modulus, order_constant, order_parameter,
                             transition_temperature, Nb2);

  LiquidCrystalElastomer::State initial_state{};

  auto qdata = solid_solver.createQuadratureDataBuffer(initial_state);

  solid_solver.setMaterial(mat, qdata);

  // prescribe zero displacement at the supported end of the beam
  std::set < int > support = {1};
  auto zero_displacement = [](const mfem::Vector&, mfem::Vector& u) -> void { u = 0.0; };
  solid_solver.setDisplacementBCs(support, zero_displacement);

  solid_solver.setDisplacement(zero_displacement);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  int num_steps = 10;

  solid_solver.outputState("paraview");

  double t    = 0.0;
  double tmax = 1.0;
  double dt   = tmax / num_steps;
  for (int i = 0; i < num_steps; i++) {
    t += dt;
    solid_solver.advanceTimestep(dt);
    solid_solver.outputState("paraview");

    temperature = initial_temperature * (1.0 - (t / tmax)) + final_temperature * (t / tmax);
  }

  MPI_Finalize();
}

// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>

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
  serac::initialize(argc, argv);

  constexpr int p   = 1;
  constexpr int dim = 3;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "lce_tensile_test_load");

  // Construct the appropriate dimension mesh and give it to the data store
  int        nElem = 2;
  double     lx = 2.5e-3, ly = 30.0e-3, lz = 30.0e-3;
  mfem::Mesh cuboid =
      mfem::Mesh(mfem::Mesh::MakeCartesian3D(nElem, 2 * nElem, 2 * nElem, mfem::Element::HEXAHEDRON, lx, ly, lz));
  auto mesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, cuboid);

  serac::StateManager::setMesh(std::move(mesh));

  double             initial_temperature = 25 + 273;
  double             final_temperature   = 430.0;
  FiniteElementState temperature(
      StateManager::newState(FiniteElementState::Options{.order = p, .name = "temperature"}));

  temperature = initial_temperature + 0.0 * final_temperature;

  FiniteElementState gamma(StateManager::newState(
      FiniteElementState::Options{.order = p, .vector_dim = 1, .element_type = ElementType::L2, .name = "gamma"}));

  int  lceArrangementTag = 1;
  auto gamma_func        = [lceArrangementTag](const mfem::Vector& x, double) -> double {
    if (lceArrangementTag == 1) {
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

  // Construct a solid mechanics solver
  LinearSolverOptions linear_options = {
      .linear_solver  = LinearSolver::GMRES,
      .preconditioner = Preconditioner::HypreAMG,
      .relative_tol   = 1.0e-6,
      .absolute_tol   = 1.0e-14,
      .max_iterations = 600,
      .print_level    = 0,
  };

#ifdef MFEM_USE_SUNDIALS
  NonlinearSolverOptions nonlinear_options = {.nonlin_solver  = serac::NonlinearSolver::KINBacktrackingLineSearch,
                                              .relative_tol   = 1.0e-4,
                                              .absolute_tol   = 1.0e-7,
                                              .max_iterations = 6,
                                              .print_level    = 1};
#else
  NonlinearSolverOptions nonlinear_options = {.nonlin_solver  = serac::NonlinearSolver::Newton,
                                              .relative_tol   = 1.0e-4,
                                              .absolute_tol   = 1.0e-7,
                                              .max_iterations = 6,
                                              .print_level    = 1};
#endif

  SolidMechanics<p, dim, Parameters<H1<p>, L2<p> > > solid_solver(nonlinear_options, linear_options,
                                                                  solid_mechanics::default_quasistatic_options,
                                                                  GeometricNonlinearities::Off, "lce_solid_functional");

  constexpr int TEMPERATURE_INDEX = 0;
  constexpr int GAMMA_INDEX       = 1;

  solid_solver.setParameter(TEMPERATURE_INDEX, temperature);
  solid_solver.setParameter(GAMMA_INDEX, gamma);

  double density                = 1.0;
  double E                      = 7.0e7;
  double nu                     = 0.45;
  double shear_modulus          = 0.5 * E / (1.0 + nu);
  double bulk_modulus           = E / 3.0 / (1.0 - 2.0 * nu);
  double order_constant         = 10;
  double order_parameter        = 0.10;
  double transition_temperature = 348;
  double Nb2                    = 1.0;

  LiquidCrystElastomerBrighenti mat(density, shear_modulus, bulk_modulus, order_constant, order_parameter,
                                    transition_temperature, Nb2);

  LiquidCrystElastomerBrighenti::State initial_state{};
  auto                                 qdata = solid_solver.createQuadratureDataBuffer(initial_state);
  solid_solver.setMaterial(DependsOn<TEMPERATURE_INDEX, GAMMA_INDEX>{}, mat, qdata);

  // prescribe symmetry conditions
  auto zeroFunc = [](const mfem::Vector /*x*/) { return 0.0; };
  solid_solver.setDisplacementBCs({1}, zeroFunc, 2);  // bottom face z-dir disp = 0
  solid_solver.setDisplacementBCs({2}, zeroFunc, 1);  // left face y-dir disp = 0
  solid_solver.setDisplacementBCs({5}, zeroFunc, 0);  // back face x-dir disp = 0

  // set initila displacement different than zero to help solver
  auto ini_displacement = [](const mfem::Vector&, mfem::Vector& u) -> void { u = 1.0e-5; };

  double iniLoadVal = 1.0e0;
  double maxLoadVal = 4 * 1.3e0 / lx / lz;
  double loadVal    = iniLoadVal + 0.0 * maxLoadVal;
  solid_solver.setPiolaTraction([&loadVal, ly](auto x, auto /*n*/, auto /*t*/) {
    return tensor<double, 3>{0, loadVal * (x[1] > 0.99 * ly), 0};
  });

  solid_solver.setDisplacement(ini_displacement);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform first quasi-static solve
  std::string output_filename = "sol_lce_brighenti_tensile";
  solid_solver.outputState(output_filename);

  // QoI for output:
  auto&                          pmesh = serac::StateManager::mesh();
  Functional<double(H1<p, dim>)> avgYDispQoI({&solid_solver.displacement().space()});
  avgYDispQoI.AddSurfaceIntegral(
      DependsOn<0>{},
      [=](auto x, auto n, auto displacement) {
        auto [u, du_dxi] = displacement;
        return dot(u, n) * ((x[1] > 0.99 * ly) ? 1.0 : 0.0);
      },
      pmesh);

  Functional<double(H1<p, dim>)> area({&solid_solver.displacement().space()});
  area.AddSurfaceIntegral(
      DependsOn<>{}, [=](auto x, auto /*n*/) { return (x[1] > 0.99 * ly) ? 1.0 : 0.0; }, pmesh);

  double initial_area = area(solid_solver.displacement());
  SLIC_INFO_ROOT("... Initial Area of the top surface: " << initial_area);

  // initializations for quasi-static problem
  int    num_steps = 3;
  double t         = 0.0;
  double tmax      = 1.0;
  double dt        = tmax / num_steps;
  double gblDispYmax;
  bool   outputDispInfo(true);

  // Perform remaining quasi-static solve
  for (int i = 0; i < (num_steps + 1); i++) {
    SLIC_INFO_ROOT(
        axom::fmt::format("\n\n............................"
                          "\n... Entering time step: {}"
                          "\n............................\n"
                          "\n... At time: {} \n... And with a tension load of: {} ( {} `%` of max)"
                          "\n... And with uniform temperature of: {}\n",
                          i + 1, t, loadVal, loadVal / maxLoadVal * 100, initial_temperature));

    // solve problem with current parameters
    solid_solver.advanceTimestep(dt);
    solid_solver.outputState(output_filename);

    // get QoI
    double current_qoi  = avgYDispQoI(solid_solver.displacement());
    double current_area = area(solid_solver.displacement());

    // get displacement info
    if (outputDispInfo) {
      auto&                 fes             = solid_solver.displacement().space();
      mfem::ParGridFunction displacement_gf = solid_solver.displacement().gridFunction();
      mfem::Vector          dispVecY(fes.GetNDofs());
      dispVecY = 0.0;

      for (int k = 0; k < fes.GetNDofs(); k++) {
        dispVecY(k) = displacement_gf(3 * k + 1);
      }

      double lclDispYmax = dispVecY.Max();
      MPI_Allreduce(&lclDispYmax, &gblDispYmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

      SLIC_INFO_ROOT(
          axom::fmt::format("\n... Max Y displacement: {}"
                            "\n... The QoIVal is: {}"
                            "\n... The top surface current area is: {}"
                            "\n... The vertical displacement integrated over the top surface is: {}",
                            gblDispYmax, current_qoi, current_area, current_qoi / current_area));
    }

    SLIC_ERROR_ROOT_IF(std::isnan(gblDispYmax), "... Solution blew up... Check boundary and initial conditions.");

    // update pseudotime-dependent information
    t += dt;
    loadVal = iniLoadVal + (maxLoadVal - iniLoadVal) * std::pow(t / tmax, 0.75);
  }

  // check output
  EXPECT_NEAR(gblDispYmax, 1.95036097e-05, 1.0e-8);

  serac::exitGracefully();
}

// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file without_input_file.cpp
 *
 * @brief A simple example of steady-state thermal conduction that uses
 * the C++ API to configure the simulation
 */

#include "serac/physics/thermal_conduction.hpp" // for serac's thermal conduction module
#include "serac/infrastructure/initialize.hpp" // for serac::initialize
#include "serac/infrastructure/terminator.hpp" // for serac::exitGracefully
#include "serac/numerics/mesh_utils.hpp" // for serac::buildRectangleMesh

int main(int argc, char* argv[])
{
  /*auto [num_procs, rank] = */serac::initialize(argc, argv);

  auto mesh = serac::buildRectangleMesh(10, 10);

  constexpr int order = 2;
  serac::ThermalConduction conduction(order, mesh, serac::ThermalConduction::defaultQuasistaticOptions());

  constexpr double kappa = 0.5;
  auto kappa_coef = std::make_unique<mfem::ConstantCoefficient>(kappa);
  conduction.setConductivity(std::move(kappa_coef));

  const std::set<int> boundary_attributes = {1};
  constexpr double boundary_value = 1.0;
  auto boundary_coef = std::make_unique<mfem::ConstantCoefficient>(boundary_value);
  conduction.setTemperatureBCs(boundary_attributes, std::move(boundary_coef));

  conduction.initializeOutput(serac::OutputType::VisIt, "simple_conduction_without_input_file");

  // Complete the solver setup
  conduction.completeSetup();
  // Output the initial state
  conduction.outputState();

  double dt; // Unused for steady-state simulations
  conduction.advanceTimestep(dt);

  // Output the final state
  conduction.outputState();

  serac::exitGracefully();
}

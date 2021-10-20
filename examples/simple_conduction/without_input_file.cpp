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

#include "serac/physics/thermal_conduction.hpp"
#include "serac/physics/utilities/state_manager.hpp"
#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/terminator.hpp"
#include "serac/numerics/mesh_utils.hpp"

int main(int argc, char* argv[])
{
  /*auto [num_procs, rank] = */serac::initialize(argc, argv);

  // Initialize the data store
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore);

  // Read the mesh options from the input file via inlet
  auto mesh = serac::mesh::refineAndDistribute(serac::buildRectangleMesh(10, 10));
  serac::StateManager::setMesh(std::move(mesh));

  // Build a thermal conduction module using the default quasi-static solver options
  constexpr int order = 2;
  serac::ThermalConduction conduction(order, serac::ThermalConduction::defaultQuasistaticOptions());

  // Build an MFEM coefficient for the thermal conductivity
  constexpr double kappa = 0.5;
  auto kappa_coef = std::make_unique<mfem::ConstantCoefficient>(kappa);
  conduction.setConductivity(std::move(kappa_coef));

  // Set a temperature (Dirichlet) boundary condition on a boundary attribute set
  const std::set<int> boundary_constant_attributes = {1};
  auto boundary_constant_coef = std::make_unique<mfem::ConstantCoefficient>(1.0);
  conduction.setTemperatureBCs(boundary_constant_attributes, std::move(boundary_constant_coef));

  // Set a different temperature boundary condition on a boundary attribute set
  const std::set<int> boundary_function_attributes = {2, 3};
  auto boundary_function_coef = std::make_unique<mfem::FunctionCoefficient>([](const mfem::Vector& vec){
    return vec[0] * vec[0] + vec[1] - 1;
  });
  conduction.setTemperatureBCs(boundary_function_attributes, std::move(boundary_function_coef));

  // Initialize the output files
  conduction.initializeOutput(serac::OutputType::ParaView, "simple_conduction_without_input_file");

  // Finalize the MFEM-based data structures inside the thermal conduction module
  conduction.completeSetup();

  // Output the initial state to the chosen file type
  conduction.outputState();

  // This solves the PDE system. As the given input file is quasi-static, this call
  // performs a single solve.
  double dt = 1.0;
  conduction.advanceTimestep(dt);

  // Output the final state
  conduction.outputState();

  // Clean up all of the software infrastructure
  serac::exitGracefully();
}
// _exit_end

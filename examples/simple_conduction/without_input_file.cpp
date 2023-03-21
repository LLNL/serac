// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
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

// _incl_thermal_header_start
#include "serac/physics/thermal_conduction_legacy.hpp"
// _incl_thermal_header_end
// _incl_state_manager_start
#include "serac/physics/state/state_manager.hpp"
// _incl_state_manager_end
// _incl_infra_start
#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/terminator.hpp"
// _incl_infra_end
// _incl_mesh_start
#include "serac/mesh/mesh_utils.hpp"
// _incl_mesh_end

// _main_init_start
int main(int argc, char* argv[])
{
  /*auto [num_procs, rank] = */serac::initialize(argc, argv);
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "without_input_file_example");
  // _main_init_end
  // _create_mesh_start
  auto mesh = serac::mesh::refineAndDistribute(serac::buildRectangleMesh(10, 10));
  serac::StateManager::setMesh(std::move(mesh));
  // _create_mesh_end

  // _create_module_start
  constexpr int order = 2;
  serac::ThermalConductionLegacy conduction(order, serac::ThermalConductionLegacy::defaultQuasistaticOptions());
  // _create_module_end

  // _conductivity_start
  constexpr double kappa = 0.5;
  auto kappa_coef = std::make_unique<mfem::ConstantCoefficient>(kappa);
  conduction.setConductivity(std::move(kappa_coef));
  // _conductivity_end
  // _bc_start
  const std::set<int> boundary_constant_attributes = {1};
  constexpr double boundary_constant = 1.0;
  auto boundary_constant_coef = std::make_unique<mfem::ConstantCoefficient>(boundary_constant);
  conduction.setTemperatureBCs(boundary_constant_attributes, std::move(boundary_constant_coef));

  const std::set<int> boundary_function_attributes = {2, 3};
  auto boundary_function_coef = std::make_unique<mfem::FunctionCoefficient>([](const mfem::Vector& vec){
    return vec[0] * vec[0] + vec[1] - 1;
  });
  conduction.setTemperatureBCs(boundary_function_attributes, std::move(boundary_function_coef));
  // _bc_end

  // _run_sim_start
  conduction.completeSetup();
  conduction.outputState();

  double dt;
  conduction.advanceTimestep(dt);
  conduction.outputState();
  // _run_sim_end

  // _exit_start
  serac::exitGracefully();
}
// _exit_end

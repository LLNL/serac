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

// _incl_thermal_header_start
#include "serac/physics/thermal_conduction.hpp"
// _incl_thermal_header_end
// _incl_infra_start
#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/terminator.hpp"
// _incl_infra_end
// _incl_mesh_start
#include "serac/numerics/mesh_utils.hpp"
// _incl_mesh_end

// _main_init_start
int main(int argc, char* argv[])
{
  /*auto [num_procs, rank] = */serac::initialize(argc, argv);
  // _main_init_end
  // _create_mesh_start
  auto mesh = serac::buildRectangleMesh(10, 10);
  // _create_mesh_end

  // _create_module_start
  constexpr int order = 2;
  serac::ThermalConduction conduction(order, mesh, serac::ThermalConduction::defaultQuasistaticOptions());
  // _create_module_end

  // _conductivity_start
  constexpr double kappa = 0.5;
  auto kappa_coef = std::make_unique<mfem::ConstantCoefficient>(kappa);
  conduction.setConductivity(std::move(kappa_coef));
  // _conductivity_end
  // _bc_start
  const std::set<int> boundary_attributes = {1};
  constexpr double boundary_value = 1.0;
  auto boundary_coef = std::make_unique<mfem::ConstantCoefficient>(boundary_value);
  conduction.setTemperatureBCs(boundary_attributes, std::move(boundary_coef));
  // _bc_end

  // _output_type_start
  conduction.initializeOutput(serac::OutputType::ParaView, "simple_conduction_without_input_file");
  // _output_type_end

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

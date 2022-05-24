// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

////////////////////////
// MECHANICAL PROBLEM //
////////////////////////

/**
 * @file without_input_file.cpp
 *
 * @brief A simple example of steady-state thermal conduction that uses
 * the C++ API to configure the simulation
 */

// _incl_thermal_header_start
// #include "serac/physics/thermal_conduction.hpp"
#include "serac/physics/lce_solid_functional.hpp"
#include "serac/physics/materials/lce_solid_functional_material.hpp"
#include "serac/physics/materials/parameterized_lce_solid_functional_material.hpp"
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
  // int p = 1, dim = 2;
  constexpr int p = 2;
  constexpr int dim = 2;

  /*auto [num_procs, rank] = */serac::initialize(argc, argv);
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "lce_mechanical");
  // _main_init_end
  // _create_mesh_start
  auto mesh = serac::mesh::refineAndDistribute(serac::buildRectangleMesh(10, 10));
  serac::StateManager::setMesh(std::move(mesh));
  // _create_mesh_end

  const std::set<int> boundary_constant_attributes = {4};

  // define the solver configurations
  const serac::IterativeSolverOptions default_linear_options = {.rel_tol     = 1.0e-6,
                                                         .abs_tol     = 1.0e-10,
                                                         .print_level = 0,
                                                         .max_iter    = 500,
                                                         .lin_solver  = serac::LinearSolver::GMRES,
                                                         .prec        = serac::HypreBoomerAMGPrec{}};

  const serac::NonlinearSolverOptions default_nonlinear_options = {
      .rel_tol = 1.0e-4, .abs_tol = 1.0e-8, .max_iter = 10, .print_level = 1};

  const typename serac::solid_util::SolverOptions default_static = {default_linear_options, default_nonlinear_options};

  // Construct a functional-based solid mechanics solver
  serac::SolidFunctional<p, dim> solid_solver(
    default_static, serac::GeometricNonlinearities::On, serac::FinalMeshOption::Reference, "solid_functional");

  serac::solid_util::NeoHookeanSolid<dim> mat(1.0, 1.0, 1.0);
  solid_solver.setMaterial(mat);

  // Define the function for the initial displacement and boundary condition
  auto bc = [](const mfem::Vector&, mfem::Vector& bc_vec) -> void { bc_vec = 0.0; };

  // Set the initial displacement and boundary condition
  solid_solver.setDisplacementBCs(boundary_constant_attributes, bc);
  solid_solver.setDisplacement(bc);

  serac::tensor<double, dim> constant_force;

  constant_force[0] = 1.0e-4;
  constant_force[1] = 2.0e-3;

  serac::solid_util::ConstantBodyForce<dim> force{constant_force};
  solid_solver.addBodyForce(force);

  solid_solver.initializeOutput(serac::OutputType::ParaView, "sol_lce_mechanical");
  // _output_type_end

  // Finalize the data structures
  solid_solver.completeSetup();
  solid_solver.outputState();

  // Perform the quasi-static solve
  double dt = 0.08;
  solid_solver.advanceTimestep(dt);
  solid_solver.outputState();

  // _exit_start
  serac::exitGracefully();

/*
  // _create_module_start
  constexpr int order = 2;
  serac::ThermalConduction conduction(order, serac::ThermalConduction::defaultQuasistaticOptions());
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

  // _output_type_start
  conduction.initializeOutput(serac::OutputType::ParaView, "simple_conduction_without_input_file_output");
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
  */
}
// _exit_end

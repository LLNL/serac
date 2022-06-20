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
#include "serac/physics/lce_mechanical_functional.hpp"
#include "serac/physics/materials/lce_mechanical_functional_material.hpp"
#include "serac/physics/materials/parameterized_lce_mechanical_functional_material.hpp"
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
  auto mesh = serac::mesh::refineAndDistribute(serac::buildRectangleMesh(20, 10, 0.002, 0.001));
  serac::StateManager::setMesh(std::move(mesh));
  // _create_mesh_end

  // define the solver configurations
  const serac::IterativeSolverOptions default_linear_options = {.rel_tol     = 1.0e-6,
                                                         .abs_tol     = 1.0e-10,
                                                         .print_level = 0,
                                                         .max_iter    = 500,
                                                         .lin_solver  = serac::LinearSolver::GMRES,
                                                         .prec        = serac::HypreBoomerAMGPrec{}};

  const serac::NonlinearSolverOptions default_nonlinear_options = {
      .rel_tol = 1.0e-4, .abs_tol = 1.0e-8, .max_iter = 10, .print_level = 1};

  const typename serac::lce_mechanical_util::SolverOptions default_static = {default_linear_options, default_nonlinear_options};

  // Construct a functional-based solid mechanics solver
  serac::LCEMechanicalFuncional<p, dim> lce_mechanical_solver(
    default_static, serac::GeometricNonlinearities::On, serac::FinalMeshOption::Reference, "mechanical_functional");

  serac::lce_mechanical_util::BrighentiMechanical<dim> lceMat(
    1.0, /*density*/
    13.33e3, /*shear_modulus*/
    10.0, /*order_constant*/
    0.46, /*order_parameter*/
    40+273, /*transition_temperature 92*/
    1e4 /*hydrostatic_pressure*/);
  lce_mechanical_solver.setMaterial(lceMat);

  // Define the function for the initial displacement and boundary condition
  auto zero_bc = [](const mfem::Vector&, mfem::Vector& bc_vec) -> void { bc_vec = 0.0; };
  // auto presc_bc = [](const mfem::Vector&, mfem::Vector& bc_vec) -> void { bc_vec = -4e-5; };
  // auto zero = std::make_shared<mfem::ConstantCoefficient>(0.0);

  // Set the initial displacement and boundary condition
  
  const std::set<int> fixed_bc_attr = {4};
  // const std::set<int> presc_bc_attr = {2};
  lce_mechanical_solver.setDisplacementBCs(fixed_bc_attr, zero_bc);
  // lce_mechanical_solver.setDisplacementBCs(presc_bc_attr, presc_bc);
  
  // lce_mechanical_solver.setDisplacementBCs({4}, zero, 0);
  // lce_mechanical_solver.setDisplacementBCs({3}, zero, 1);
  
  lce_mechanical_solver.setDisplacement(zero_bc);

  serac::lce_mechanical_util::TractionFunction<dim> traction_function{
      [](const serac::tensor<double, dim>& x, const serac::tensor<double, dim>&, const double) {
        serac::tensor<double, dim> traction;
        for (int i = 0; i < dim; ++i) {
          traction[i] = 0.0;
        }

        if (x[0] > 1.9e-3) {
          traction[1] = 0.0; // -5.0e-3;
        }
        return traction;
      }};

  lce_mechanical_solver.setTractionBCs(traction_function);
  // serac::tensor<double, dim> constant_force;
  // constant_force[0] = -1.0e-5; // 1.0e-4;
  // constant_force[1] = 0.0; // 2.0e-3;

  // serac::lce_mechanical_util::ConstantBodyForce<dim> force{constant_force};
  // lce_mechanical_solver.addBodyForce(force);

  lce_mechanical_solver.initializeOutput(serac::OutputType::ParaView, "sol_lce_mechanical");
  // _output_type_end

  // Finalize the data structures
  lce_mechanical_solver.completeSetup();
  lce_mechanical_solver.outputState();

  // Perform the quasi-static solve
  double dt = 0.08;
  lce_mechanical_solver.advanceTimestep(dt);
  lce_mechanical_solver.outputState();

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
  const std::set<int> fixed_bc_attr = {1};
  constexpr double boundary_constant = 1.0;
  auto boundary_constant_coef = std::make_unique<mfem::ConstantCoefficient>(boundary_constant);
  conduction.setTemperatureBCs(fixed_bc_attr, std::move(boundary_constant_coef));

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

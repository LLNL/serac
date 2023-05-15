// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/heat_transfer_input.hpp"

namespace serac {

void HeatTransferInputOptions::defineInputFileSchema(axom::inlet::Container& container)
{
  // Polynomial interpolation order - currently up to 8th order is allowed
  container.addInt("order", "Order degree of the finite elements.").defaultValue(1).range(1, 8);

  // material parameters
  container.addDouble("kappa", "Thermal conductivity").defaultValue(0.5);
  container.addDouble("rho", "Density").defaultValue(1.0);
  container.addDouble("cp", "Specific heat capacity").defaultValue(1.0);

  auto& source = container.addStruct("source", "Scalar source term (RHS of the thermal conduction PDE)");
  serac::input::CoefficientInputOptions::defineInputFileSchema(source);

  auto& equation_solver_container =
      container.addStruct("equation_solver", "Linear and Nonlinear stiffness Solver Parameters.");
  serac::EquationSolver::defineInputFileSchema(equation_solver_container);

  auto& dynamics_container = container.addStruct("dynamics", "Parameters for mass matrix inversion");
  dynamics_container.addString("timestepper", "Timestepper (ODE) method to use");
  dynamics_container.addString("enforcement_method", "Time-varying constraint enforcement method to use");

  auto& bc_container = container.addStructDictionary("boundary_conds", "Container of boundary conditions");
  serac::input::BoundaryConditionInputOptions::defineInputFileSchema(bc_container);

  auto& init_temp = container.addStruct("initial_temperature", "Coefficient for initial condition");
  serac::input::CoefficientInputOptions::defineInputFileSchema(init_temp);
}

}  // namespace serac

serac::HeatTransferInputOptions FromInlet<serac::HeatTransferInputOptions>::operator()(
    const axom::inlet::Container& base)
{
  serac::HeatTransferInputOptions result;

  result.order = base["order"];

  // Solver parameters
  auto equation_solver         = base["equation_solver"];
  result.lin_solver_options    = equation_solver["linear"].get<serac::LinearSolverOptions>();
  result.nonlin_solver_options = equation_solver["nonlinear"].get<serac::NonlinearSolverOptions>();

  if (base.contains("dynamics")) {
    serac::TimesteppingOptions timestepping_options;
    auto                       dynamics = base["dynamics"];

    // FIXME: Implement all supported methods as part of an ODE schema
    const static std::map<std::string, serac::TimestepMethod> timestep_methods = {
        {"AverageAcceleration", serac::TimestepMethod::AverageAcceleration},
        {"BackwardEuler", serac::TimestepMethod::BackwardEuler},
        {"ForwardEuler", serac::TimestepMethod::ForwardEuler}};
    std::string timestep_method = dynamics["timestepper"];
    SLIC_ERROR_ROOT_IF(timestep_methods.count(timestep_method) == 0,
                       "Unrecognized timestep method: " << timestep_method);
    timestepping_options.timestepper = timestep_methods.at(timestep_method);

    // FIXME: Implement all supported methods as part of an ODE schema
    const static std::map<std::string, serac::DirichletEnforcementMethod> enforcement_methods = {
        {"RateControl", serac::DirichletEnforcementMethod::RateControl}};
    std::string enforcement_method = dynamics["enforcement_method"];
    SLIC_ERROR_ROOT_IF(enforcement_methods.count(enforcement_method) == 0,
                       "Unrecognized enforcement method: " << enforcement_method);
    timestepping_options.enforcement_method = enforcement_methods.at(enforcement_method);

    result.timestepping_options = timestepping_options;
  }

  if (base.contains("source")) {
    result.source_coef = base["source"].get<serac::input::CoefficientInputOptions>();
  }

  // Set the material parameters
  result.kappa = base["kappa"];
  result.rho   = base["rho"];
  result.cp    = base["cp"];

  result.boundary_conditions =
      base["boundary_conds"].get<std::unordered_map<std::string, serac::input::BoundaryConditionInputOptions>>();

  if (base.contains("initial_temperature")) {
    result.initial_temperature = base["initial_temperature"].get<serac::input::CoefficientInputOptions>();
  }
  return result;
}

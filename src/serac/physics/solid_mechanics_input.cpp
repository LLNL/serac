// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_mechanics_input.hpp"

namespace serac {

void SolidMechanicsInputOptions::defineInputFileSchema(axom::inlet::Container& container)
{
  // interpolation order - currently up to 3rd order is allowed
  container.addInt("order", "polynomial order of the basis functions.").defaultValue(1).range(1, 3);

  auto& material_container = container.addStructArray("materials", "Container for array of materials");
  SolidMaterialInputOptions::defineInputFileSchema(material_container);

  // Geometric nonlinearities flag
  container.addBool("geometric_nonlin", "Flag to include geometric nonlinearities in the residual calculation.")
      .defaultValue(true);

  auto& equation_solver_container =
      container.addStruct("equation_solver", "Linear and Nonlinear stiffness Solver Parameters.");
  EquationSolver::defineInputFileSchema(equation_solver_container);

  auto& dynamics_container = container.addStruct("dynamics", "Parameters for mass matrix inversion");
  dynamics_container.addString("timestepper", "Timestepper (ODE) method to use");
  dynamics_container.addString("enforcement_method", "Time-varying constraint enforcement method to use");

  auto& bc_container = container.addStructDictionary("boundary_conds", "Container of boundary conditions");
  input::BoundaryConditionInputOptions::defineInputFileSchema(bc_container);

  auto& init_displ = container.addStruct("initial_displacement", "Coefficient for initial condition");
  input::CoefficientInputOptions::defineInputFileSchema(init_displ);
  auto& init_velo = container.addStruct("initial_velocity", "Coefficient for initial condition");
  input::CoefficientInputOptions::defineInputFileSchema(init_velo);
}

}  // namespace serac

serac::SolidMechanicsInputOptions FromInlet<serac::SolidMechanicsInputOptions>::operator()(
    const axom::inlet::Container& base)
{
  serac::SolidMechanicsInputOptions result;

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
        {"NewmarkBeta", serac::TimestepMethod::Newmark},
        {"BackwardEuler", serac::TimestepMethod::BackwardEuler}};
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

    result.timestepping_options = std::move(timestepping_options);
  }

  result.materials = base["materials"].get<std::vector<serac::var_solid_material_t>>();

  // Set the geometric nonlinearities flag
  bool input_geom_nonlin = base["geometric_nonlin"];
  if (input_geom_nonlin) {
    result.geom_nonlin = serac::GeometricNonlinearities::On;
  } else {
    result.geom_nonlin = serac::GeometricNonlinearities::Off;
  }

  if (base.contains("boundary_conds")) {
    result.boundary_conditions =
        base["boundary_conds"].get<std::unordered_map<std::string, serac::input::BoundaryConditionInputOptions>>();
  }

  if (base.contains("initial_displacement")) {
    result.initial_displacement = base["initial_displacement"].get<serac::input::CoefficientInputOptions>();
  }
  if (base.contains("initial_velocity")) {
    result.initial_velocity = base["initial_velocity"].get<serac::input::CoefficientInputOptions>();
  }
  return result;
}

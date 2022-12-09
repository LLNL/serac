// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_mechanics_input.hpp"

namespace serac {

void SolidMechanicsInputOptions::defineInputFileSchema(axom::inlet::Container& container)
{
  // Polynomial interpolation order - currently up to 8th order is allowed
  container.addInt("order", "Order degree of the finite elements.").defaultValue(1).range(1, 8);

  // neo-Hookean material parameters
  container.addDouble("mu", "Shear modulus in the Neo-Hookean hyperelastic model.").defaultValue(0.25);
  container.addDouble("K", "Bulk modulus in the Neo-Hookean hyperelastic model.").defaultValue(5.0);

  // Geometric nonlinearities flag
  container.addBool("geometric_nonlin", "Flag to include geometric nonlinearities in the residual calculation.")
      .defaultValue(true);

  // Geometric nonlinearities flag
  container
      .addBool("material_nonlin",
               "Flag to include material nonlinearities (linear elastic vs. neo-Hookean material model).")
      .defaultValue(true);

  container.addDouble("viscosity", "Viscosity constant").defaultValue(0.0);

  container.addDouble("density", "Initial mass density").defaultValue(1.0);

  auto& equation_solver_container =
      container.addStruct("equation_solver", "Linear and Nonlinear stiffness Solver Parameters.");
  serac::mfem_ext::EquationSolver::DefineInputFileSchema(equation_solver_container);

  auto& dynamics_container = container.addStruct("dynamics", "Parameters for mass matrix inversion");
  dynamics_container.addString("timestepper", "Timestepper (ODE) method to use");
  dynamics_container.addString("enforcement_method", "Time-varying constraint enforcement method to use");

  auto& bc_container = container.addStructDictionary("boundary_conds", "Container of boundary conditions");
  serac::input::BoundaryConditionInputOptions::defineInputFileSchema(bc_container);

  auto& init_displ = container.addStruct("initial_displacement", "Coefficient for initial condition");
  serac::input::CoefficientInputOptions::defineInputFileSchema(init_displ);
  auto& init_velo = container.addStruct("initial_velocity", "Coefficient for initial condition");
  serac::input::CoefficientInputOptions::defineInputFileSchema(init_velo);
}

}  // namespace serac

serac::SolidMechanicsInputOptions FromInlet<serac::SolidMechanicsInputOptions>::operator()(
    const axom::inlet::Container& base)
{
  serac::SolidMechanicsInputOptions result;

  result.order = base["order"];

  // Solver parameters
  auto equation_solver            = base["equation_solver"];
  result.solver_options.linear    = equation_solver["linear"].get<serac::LinearSolverOptions>();
  result.solver_options.nonlinear = equation_solver["nonlinear"].get<serac::NonlinearSolverOptions>();

  if (base.contains("dynamics")) {
    serac::TimesteppingOptions dyn_options;
    auto                       dynamics = base["dynamics"];

    // FIXME: Implement all supported methods as part of an ODE schema
    const static std::map<std::string, serac::TimestepMethod> timestep_methods = {
        {"AverageAcceleration", serac::TimestepMethod::AverageAcceleration},
        {"NewmarkBeta", serac::TimestepMethod::Newmark},
        {"BackwardEuler", serac::TimestepMethod::BackwardEuler}};
    std::string timestep_method = dynamics["timestepper"];
    SLIC_ERROR_ROOT_IF(timestep_methods.count(timestep_method) == 0,
                       "Unrecognized timestep method: " << timestep_method);
    dyn_options.timestepper = timestep_methods.at(timestep_method);

    // FIXME: Implement all supported methods as part of an ODE schema
    const static std::map<std::string, serac::DirichletEnforcementMethod> enforcement_methods = {
        {"RateControl", serac::DirichletEnforcementMethod::RateControl}};
    std::string enforcement_method = dynamics["enforcement_method"];
    SLIC_ERROR_ROOT_IF(enforcement_methods.count(enforcement_method) == 0,
                       "Unrecognized enforcement method: " << enforcement_method);
    dyn_options.enforcement_method = enforcement_methods.at(enforcement_method);

    result.solver_options.dynamic = std::move(dyn_options);
  }

  // Set the material parameters
  // neo-Hookean material parameters
  result.mu = base["mu"];
  result.K  = base["K"];

  // Set the geometric nonlinearities flag
  bool input_geom_nonlin = base["geometric_nonlin"];
  if (input_geom_nonlin) {
    result.geom_nonlin = serac::GeometricNonlinearities::On;
  } else {
    result.geom_nonlin = serac::GeometricNonlinearities::Off;
  }

  // Set the material nonlinearity flag
  result.material_nonlin = base["material_nonlin"];

  if (base.contains("boundary_conds")) {
    result.boundary_conditions =
        base["boundary_conds"].get<std::unordered_map<std::string, serac::input::BoundaryConditionInputOptions>>();
  }

  result.viscosity = base["viscosity"];

  result.initial_mass_density = base["density"];

  if (base.contains("initial_displacement")) {
    result.initial_displacement = base["initial_displacement"].get<serac::input::CoefficientInputOptions>();
  }
  if (base.contains("initial_velocity")) {
    result.initial_velocity = base["initial_velocity"].get<serac::input::CoefficientInputOptions>();
  }
  return result;
}

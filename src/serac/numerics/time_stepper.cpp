// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_mechanics.hpp"
#include "serac/numerics/odes.hpp"
#include "serac/numerics/equation_solver.hpp"
#include "serac/physics/boundary_conditions/boundary_condition_manager.hpp"

namespace serac {

SecondOrderTimeStepper::SecondOrderTimeStepper(EquationSolver*                   solver,
                                               const serac::TimesteppingOptions& timestepping_opts)
    : nonlinear_solver(solver), timestepping_options(timestepping_opts)
{
}

void SecondOrderTimeStepper::set_states(const std::vector<FiniteElementState*>& inputStates,
                                        const std::vector<FiniteElementState*>& outputStates, 
                                        BoundaryConditionManager& bcs)
{
  SLIC_ERROR_ROOT_IF(
    inputStates.size() != 2,
    axom::fmt::format("Second order time integrator must have exactly 2 states, u, u_dot, u_dot_dot."));
  SLIC_ERROR_ROOT_IF(
    outputStates.size() != 1,
    axom::fmt::format("Second order time integrator must have exactly 2 states, u, u_dot, u_dot_dot."));
  SLIC_ERROR_ROOT_IF(
    inputStates[0]->space().TrueVSize() != inputStates[1]->space().TrueVSize(),
    axom::fmt::format("Second order time integrator states must have same true size."));
  SLIC_ERROR_ROOT_IF(
    inputStates[0]->space().TrueVSize() != outputStates[0]->space().TrueVSize(),
    axom::fmt::format("Second order time integrator states must have same true size."));

  inputStates_ = inputStates;
  outputStates_ = outputStates;

  int true_size = inputStates[0]->space().TrueVSize()
  u_.SetSize(true_size);
  v_.SetSize(true_size);
  ode2_ = std::make_unique<mfem_ext::SecondOrderODE>(
      states[0]->space().TrueVSize(),
      mfem_ext::SecondOrderODE::State{.time = ode_time_point, .c0 = c0, .c1 = c1, .u = u, .du_dt = v, .d2u_dt2 = *outputStates[0]},
      *nonlinear_solver, bcs);
  ode2->SetTimestepper(timestepping_options.timestepper);
  ode2->SetEnforcementMethod(timestepping_options.enforcement_method);
}

void SecondOrderTimeStepper::reset()
{
  u              = 0.0;
  v              = 0.0;
  c0             = 0.0;
  c1             = 0.0;
  ode_time_point = 0.0;
}

void SecondOrderTimeStepper::advance(double t, double dt)
{
  ode2->Step(u, v, t, dt);
  // printf("%g %g\n", t, dt);
}

void SecondOrderTimeStepper::reverse_vjp(double t, double dt) { printf("%g %g\n", t, dt); }

QuasiStaticStepper::QuasiStaticStepper(serac::EquationSolver*            solver,
                                       const serac::TimesteppingOptions& timestepping_opts)
    : nonlinear_solver(solver), timestepping_options(timestepping_opts)
{
}

void QuasiStaticStepper::set_states(const std::vector<FiniteElementState*>& states, BoundaryConditionManager&)
{
  SLIC_ERROR_ROOT_IF(
      states.size() != 3,
      axom::fmt::format("Second order time integrators must have exactly 3 states, u, u_dot, u_dot_dot."));
}

void QuasiStaticStepper::reset()
{
  u              = 0.0;
  v              = 0.0;
  a              = 0.0;
  c0             = 0.0;
  c1             = 0.0;
  ode_time_point = 0.0;
}

void QuasiStaticStepper::advance(double t, double dt)
{
  // ode2->Step(u, v, t, dt);
  printf("%g %g\n", t, dt);
}

void QuasiStaticStepper::reverse_vjp(double t, double dt) { printf("%g %g\n", t, dt); }

}  // namespace serac
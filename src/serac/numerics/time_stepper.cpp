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

SecondOrderIntegrator::SecondOrderIntegrator(std::unique_ptr<serac::EquationSolver> solver, const serac::TimesteppingOptions& timestepping_opts) 
  : nonlinear_solver(std::move(solver))
  , timestepping_options(timestepping_opts) {
}

void SecondOrderIntegrator::set_states(const std::vector<FiniteElementState*>& states, BoundaryConditionManager& bcs) 
{
  SLIC_ERROR_ROOT_IF(states.size() != 3,
                     axom::fmt::format("Second order time integrators must have exactly 3 states, u, u_dot, u_dot_dot."));
  ode2 = std::make_unique<mfem_ext::SecondOrderODE> (states[0]->space().TrueVSize(),
              mfem_ext::SecondOrderODE::State{.time = ode_time_point, .c0 = c0, .c1 = c1, .u = u, .du_dt = v, .d2u_dt2 = a},
              *nonlinear_solver, bcs);
  ode2->SetTimestepper(timestepping_options.timestepper);
  ode2->SetEnforcementMethod(timestepping_options.enforcement_method);
}

void SecondOrderIntegrator::reset()
{
  u = 0.0;
  v = 0.0;
  a = 0.0;
  c0 = 0.0;
  c1 = 0.0;
  ode_time_point = 0.0;
}

void advance(double t, double dt)
{
  printf("%g %g\n", t, dt);
}

void reverse_vjp(double t, double dt)
{
  printf("%g %g\n", t, dt);
}

}
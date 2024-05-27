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
  : TimeStepper(solver), timestepping_options_(timestepping_opts)
{
}

void SecondOrderTimeStepper::setStates(const FieldVec& independentStates,
                                       const FieldVec& states,
                                       BoundaryConditionManager& bcs)
{

  SLIC_ERROR_ROOT_IF(
      independentStates.size() != 2,
      axom::fmt::format("Second order time integrator must have exactly 2 input states, u, u_dot."));
  SLIC_ERROR_ROOT_IF(
      states.size() != 1,
      axom::fmt::format("Second order time integrator must have exactly 1 output state, u_dot_dot."));
  SLIC_ERROR_ROOT_IF(independentStates[0]->space().TrueVSize() != independentStates[1]->space().TrueVSize(),
                     axom::fmt::format("Second order time integrator states must have same true size."));
  SLIC_ERROR_ROOT_IF(independentStates[0]->space().TrueVSize() != states[0]->space().TrueVSize(),
                     axom::fmt::format("Second order time integrator states must have same true size."));

  independentStates_  = independentStates;
  states_ = states;
  bcManagerPtr_ = &bcs;

  true_size_ = independentStates_[0]->space().TrueVSize();
  u_.SetSize(true_size_);
  v_.SetSize(true_size_);
  u_pred_.SetSize(true_size_);
  ode2_ = std::make_unique<mfem_ext::SecondOrderODE>(
      true_size_,
      mfem_ext::SecondOrderODE::State{
          .time = ode_time_point_, .c0 = c0_, .c1 = c1_, .u = u_, .du_dt = v_, .d2u_dt2 = *states_[0]},
      *nonlinear_solver_, bcs);
  ode2_->SetTimestepper(timestepping_options_.timestepper);
  ode2_->SetEnforcementMethod(timestepping_options_.enforcement_method);
}

void SecondOrderTimeStepper::reset()
{
  u_              = 0.0;
  u_pred_         = 0.0;
  v_              = 0.0;
  c0_             = 0.0;
  c1_             = 0.0;
  ode_time_point_ = 0.0;
}

void SecondOrderTimeStepper::step(double t, double dt)
{
  std::cout << "sizs = " << independentStates_[0]->space().TrueVSize() << " "  << independentStates_[1]->space().TrueVSize() << std::endl;
  printf("steo before\n");
  ode2_->Step(*independentStates_[0], *independentStates_[1], t, dt);
  printf("steo after\n");
}

void SecondOrderTimeStepper::vjpStep(double t, double dt) { printf("%g %g\n", t, dt); }

void SecondOrderTimeStepper::finalizeFuncs() {
  auto residual_with_bcs_ = std::make_unique<mfem_ext::StdFunctionOperator>(
    true_size_,
    [this](const mfem::Vector& d2u_dt2, mfem::Vector& r) {
      printf("a1\n");
      add(1.0, u_, c0_, d2u_dt2, u_pred_);
      residual_func_(ode_time_point_,
                         TimeStepper::VectorVec{&u_pred_, &v_},
                         TimeStepper::ConstVectorVec{&d2u_dt2}, r);
      r.SetSubVector(bcManagerPtr_->allEssentialTrueDofs(), 0.0);
      printf("a2\n");
    },
    [this](const mfem::Vector& d2u_dt2) -> mfem::Operator& {
      printf("a3\n");
      add(1.0, u_, c0_, d2u_dt2, u_pred_);
      jacobian_func_(ode_time_point_,
                         TimeStepper::VectorVec{&u_pred_, &v_},
                         TimeStepper::ConstVectorVec{&d2u_dt2}, J_);
      printf("a4\n");
      return *J_;
    }
  );
  
  nonlinear_solver_->setOperator(*residual_with_bcs_);
}

QuasiStaticStepper::QuasiStaticStepper(serac::EquationSolver*            solver,
                                       const serac::TimesteppingOptions&)
    : TimeStepper(solver)
{
}

void QuasiStaticStepper::setStates(const FieldVec& independentStates,
                                   const FieldVec& states,
                                   BoundaryConditionManager&)
{
  SLIC_ERROR_ROOT_IF(
      independentStates.size() != 0,
      axom::fmt::format("Quasi-static steppers have 0 independent state."));
  SLIC_ERROR_ROOT_IF(
      states.size() != 1,
      axom::fmt::format("Quasi-static steppers have 1 state."));
}

void QuasiStaticStepper::reset()
{
  u_              = 0.0;
  ode_time_point_ = 0.0;
}

void QuasiStaticStepper::step(double t, double dt)
{
  // ode2->Step(u, v, t, dt);
  printf("%g %g\n", t, dt);
}

void QuasiStaticStepper::vjpStep(double t, double dt) { printf("%g %g\n", t, dt); }

}  // namespace serac
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

mfem::Solver& TimeStepper::linearSolver() { return nonlinear_solver_->linearSolver(); }

std::pair<const mfem::HypreParMatrix&, const mfem::HypreParMatrix&> TimeStepper::stiffnessMatrix() const
{
  SLIC_ERROR_ROOT_IF(!J_ || !J_e_, "Stiffness matrix has not yet been assembled.");
  return {*J_, *J_e_};
}

SecondOrderTimeStepper::SecondOrderTimeStepper(std::unique_ptr<EquationSolver>   nonlinear_solver,
                                               const serac::TimesteppingOptions& timestepping_opts)
    : TimeStepper(std::move(nonlinear_solver)), timestepping_options_(timestepping_opts)
{
  reset();
}

void SecondOrderTimeStepper::setStates(const FieldVec& independentStates, const FieldVec& states,
                                       BoundaryConditionManager& bcs)
{
  SLIC_ERROR_ROOT_IF(independentStates.size() != 2,
                     axom::fmt::format("Second order time integrator must have exactly 2 input states, u, u_dot."));
  SLIC_ERROR_ROOT_IF(states.size() != 1,
                     axom::fmt::format("Second order time integrator must have exactly 1 output state, u_dot_dot."));
  SLIC_ERROR_ROOT_IF(independentStates[0]->space().TrueVSize() != independentStates[1]->space().TrueVSize(),
                     axom::fmt::format("Second order time integrator states must have same true size."));
  SLIC_ERROR_ROOT_IF(independentStates[0]->space().TrueVSize() != states[0]->space().TrueVSize(),
                     axom::fmt::format("Second order time integrator states must have same true size."));

  independentStates_ = independentStates;
  states_            = states;
  bcManagerPtr_      = &bcs;

  true_size_ = states_[0]->space().TrueVSize();
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
  ode2_->Step(*independentStates_[0], *independentStates_[1], t, dt);
}

void SecondOrderTimeStepper::vjpStep(double t, double dt) { printf("%g %g\n", t, dt); }

void SecondOrderTimeStepper::finalizeFuncs()
{
  residual_with_bcs_ = std::make_unique<mfem_ext::StdFunctionOperator>(
      true_size_,
      [this](const mfem::Vector& d2u_dt2, mfem::Vector& r) {
        add(1.0, u_, c0_, d2u_dt2, u_pred_);
        residual_func_(ode_time_point_, TimeStepper::VectorVec{&u_pred_, &v_}, TimeStepper::ConstVectorVec{&d2u_dt2},
                       r);
        r.SetSubVector(bcManagerPtr_->allEssentialTrueDofs(), 0.0);
      },
      [this](const mfem::Vector& d2u_dt2) -> mfem::Operator& {
        add(1.0, u_, c0_, d2u_dt2, u_pred_);
        jacobian_func_(ode_time_point_, c0_, TimeStepper::VectorVec{&u_pred_, &v_},
                       TimeStepper::ConstVectorVec{&d2u_dt2}, J_);
        J_e_ = bcManagerPtr_->eliminateAllEssentialDofsFromMatrix(*J_);
        return *J_;
      });

  nonlinear_solver_->setOperator(*residual_with_bcs_);
}

QuasiStaticStepper::QuasiStaticStepper(std::unique_ptr<EquationSolver> nonlinear_solver,
                                       const serac::TimesteppingOptions&)
    : TimeStepper(std::move(nonlinear_solver))
{
}

void QuasiStaticStepper::setStates(const FieldVec& independentStates, const FieldVec& states,
                                   BoundaryConditionManager& bcs)
{
  SLIC_ERROR_ROOT_IF(independentStates.size() != 1,
                     axom::fmt::format("Quasi-static steppers have 1 independent state."));
  SLIC_ERROR_ROOT_IF(states.size() != 1, axom::fmt::format("Quasi-static steppers have 1 state."));

  independentStates_ = independentStates;
  states_            = states;
  bcManagerPtr_      = &bcs;

  true_size_ = states[0]->space().TrueVSize();
}

void QuasiStaticStepper::reset() { ode_time_point_ = 0.0; }

void QuasiStaticStepper::step(double time, double dt)
{
  ode_time_point_ = time + dt;
  nonlinear_solver_->solve(*states_[0]);
}

void QuasiStaticStepper::vjpStep(double t, double dt) { printf("%g %g\n", t, dt); }

void QuasiStaticStepper::finalizeFuncs()
{
  residual_with_bcs_ = std::make_unique<mfem_ext::StdFunctionOperator>(
      true_size_,
      [this](const mfem::Vector& u, mfem::Vector& r) {
        residual_func_(ode_time_point_, TimeStepper::VectorVec{independentStates_[0]}, TimeStepper::ConstVectorVec{&u},
                       r);
        r.SetSubVector(bcManagerPtr_->allEssentialTrueDofs(), 0.0);
      },
      [this](const mfem::Vector& u) -> mfem::Operator& {
        jacobian_func_(ode_time_point_, 1.0, TimeStepper::VectorVec{independentStates_[0]},
                       TimeStepper::ConstVectorVec{&u}, J_);
        J_e_ = bcManagerPtr_->eliminateAllEssentialDofsFromMatrix(*J_);
        return *J_;
      });
  nonlinear_solver_->setOperator(*residual_with_bcs_);
}

}  // namespace serac
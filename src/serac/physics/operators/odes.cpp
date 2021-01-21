// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/operators/odes.hpp"

#include "serac/numerics/expr_template_ops.hpp"

namespace serac::mfem_ext {

SecondOrderODE::SecondOrderODE(int n, State&& state, const EquationSolver& solver, const BoundaryConditionManager& bcs)
    : mfem::SecondOrderTimeDependentOperator(n, 0.0), state_(std::move(state)), solver_(solver), bcs_(bcs), zero_(n)
{
  zero_ = 0.0;
  U_minus_.SetSize(n);
  U_.SetSize(n);
  U_plus_.SetSize(n);
  dU_dt_.SetSize(n);
  d2U_dt2_.SetSize(n);
}

void SecondOrderODE::SetTimestepper(const serac::TimestepMethod timestepper)
{
  switch (timestepper) {
    case serac::TimestepMethod::HHTAlpha:
      ode_solver_ = std::make_unique<mfem::HHTAlphaSolver>();
      break;
    case serac::TimestepMethod::WBZAlpha:
      ode_solver_ = std::make_unique<mfem::WBZAlphaSolver>();
      break;
    case serac::TimestepMethod::AverageAcceleration:
      ode_solver_ = std::make_unique<mfem::AverageAccelerationSolver>();
      break;
    case serac::TimestepMethod::LinearAcceleration:
      ode_solver_ = std::make_unique<mfem::LinearAccelerationSolver>();
      break;
    case serac::TimestepMethod::CentralDifference:
      ode_solver_ = std::make_unique<mfem::CentralDifferenceSolver>();
      break;
    case serac::TimestepMethod::FoxGoodwin:
      ode_solver_ = std::make_unique<mfem::FoxGoodwinSolver>();
      break;
    default:
      SLIC_ERROR("Timestep method was not a supported second-order ODE method");
  }
  ode_solver_->Init(*this);
}

void SecondOrderODE::Solve(const double t, const double c0, const double c1, const mfem::Vector& u,
                           const mfem::Vector& du_dt, mfem::Vector& d2u_dt2) const
{
  // this is intended to be temporary
  // Ideally, epsilon should be "small" relative to the characteristic time
  // of the ODE, but we can't ensure that at present (we don't have a
  // critical timestep estimate)
  constexpr double epsilon = 0.0001;

  // assign these values to variables with greater scope,
  // so that the residual operator can see them
  state_.c0    = c0;
  state_.c1    = c1;
  state_.u     = u;
  state_.du_dt = du_dt;

  // TODO: take care of this last part of the ODE definition
  //       automatically by wrapping mfem's ODE solvers
  //
  // evaluate the constraint functions at a 3-point
  // stencil of times centered on the time of interest
  // in order to compute finite-difference approximations
  // to the time derivatives that appear in the residual
  U_minus_ = 0.0;
  U_       = 0.0;
  U_plus_  = 0.0;
  for (const auto& bc : bcs_.essentials()) {
    bc.projectBdrToDofs(U_minus_, t - epsilon);
    bc.projectBdrToDofs(U_, t);
    bc.projectBdrToDofs(U_plus_, t + epsilon);
  }

  bool implicit = (c0 != 0.0 || c1 != 0.0);
  if (implicit) {
    if (enforcement_method_ == DirichletEnforcementMethod::DirectControl) {
      d2U_dt2_ = (U_ - u) / c0;
      dU_dt_   = du_dt;
      U_       = u;
    }

    if (enforcement_method_ == DirichletEnforcementMethod::RateControl) {
      d2U_dt2_ = (dU_dt_ - du_dt) / c1;
      dU_dt_   = du_dt;
      U_       = u;
    }

    if (enforcement_method_ == DirichletEnforcementMethod::FullControl) {
      d2U_dt2_ = (U_minus_ - 2.0 * U_ + U_plus_) / (epsilon * epsilon);
      dU_dt_   = (U_plus_ - U_minus_) / (2.0 * epsilon) - c1 * d2U_dt2_;
      U_       = U_ - c0 * d2U_dt2_;
    }
  } else {
    d2U_dt2_ = (U_minus_ - 2.0 * U_ + U_plus_) / (epsilon * epsilon);
    dU_dt_   = (U_plus_ - U_minus_) / (2.0 * epsilon);
  }

  auto constrained_dofs = bcs_.allEssentialDofs();
  state_.u.SetSubVector(constrained_dofs, 0.0);
  U_.SetSubVectorComplement(constrained_dofs, 0.0);
  state_.u += U_;

  state_.du_dt.SetSubVector(constrained_dofs, 0.0);
  dU_dt_.SetSubVectorComplement(constrained_dofs, 0.0);
  state_.du_dt += dU_dt_;

  // use the previous solution as our starting guess
  d2u_dt2 = state_.d2u_dt2;
  d2u_dt2.SetSubVector(constrained_dofs, 0.0);
  d2U_dt2_.SetSubVectorComplement(constrained_dofs, 0.0);
  d2u_dt2 += d2U_dt2_;

  solver_.Mult(zero_, d2u_dt2);
  SLIC_WARNING_IF(!solver_.NonlinearSolver().GetConverged(), "Newton Solver did not converge.");

  state_.d2u_dt2 = d2u_dt2;
}

FirstOrderODE::FirstOrderODE(int n, State&& state, const EquationSolver& solver, const BoundaryConditionManager& bcs)
    : mfem::TimeDependentOperator(n, 0.0), state_(std::move(state)), solver_(solver), bcs_(bcs), zero_(n)
{
  zero_ = 0.0;
  U_minus_.SetSize(n);
  U_.SetSize(n);
  U_plus_.SetSize(n);
  dU_dt_.SetSize(n);
}

void FirstOrderODE::SetTimestepper(const serac::TimestepMethod timestepper)
{
  switch (timestepper) {
    case serac::TimestepMethod::BackwardEuler:
      ode_solver_ = std::make_unique<mfem::BackwardEulerSolver>();
      break;
    case serac::TimestepMethod::SDIRK33:
      ode_solver_ = std::make_unique<mfem::SDIRK33Solver>();
      break;
    case serac::TimestepMethod::ForwardEuler:
      ode_solver_ = std::make_unique<mfem::ForwardEulerSolver>();
      break;
    case serac::TimestepMethod::RK2:
      ode_solver_ = std::make_unique<mfem::RK2Solver>(0.5);
      break;
    case serac::TimestepMethod::RK3SSP:
      ode_solver_ = std::make_unique<mfem::RK3SSPSolver>();
      break;
    case serac::TimestepMethod::RK4:
      ode_solver_ = std::make_unique<mfem::RK4Solver>();
      break;
    case serac::TimestepMethod::GeneralizedAlpha:
      ode_solver_ = std::make_unique<mfem::GeneralizedAlphaSolver>(0.5);
      break;
    case serac::TimestepMethod::ImplicitMidpoint:
      ode_solver_ = std::make_unique<mfem::ImplicitMidpointSolver>();
      break;
    case serac::TimestepMethod::SDIRK23:
      ode_solver_ = std::make_unique<mfem::SDIRK23Solver>();
      break;
    case serac::TimestepMethod::SDIRK34:
      ode_solver_ = std::make_unique<mfem::SDIRK34Solver>();
      break;
    default:
      SLIC_ERROR("Timestep method was not a supported first-order ODE method");
  }
  ode_solver_->Init(*this);
}

void FirstOrderODE::Solve(const double dt, const mfem::Vector& u, mfem::Vector& du_dt) const
{
  // this is intended to be temporary
  // Ideally, epsilon should be "small" relative to the characteristic
  // time of the ODE, but we can't ensure that at present (we don't have
  // a critical timestep estimate)
  constexpr double epsilon = 0.0001;

  // assign these values to variables with greater scope,
  // so that the residual operator can see them
  state_.dt = dt;
  state_.u  = u;

  // TODO: take care of this last part of the ODE definition
  //       automatically by wrapping mfem's ODE solvers
  //
  // evaluate the constraint functions at a 3-point
  // stencil of times centered on the time of interest
  // in order to compute finite-difference approximations
  // to the time derivatives that appear in the residual
  U_minus_ = 0.0;
  U_       = 0.0;
  U_plus_  = 0.0;
  for (const auto& bc : bcs_.essentials()) {
    bc.projectBdrToDofs(U_minus_, t - epsilon);
    bc.projectBdrToDofs(U_, t);
    bc.projectBdrToDofs(U_plus_, t + epsilon);
  }

  bool implicit = (dt != 0.0);
  if (implicit) {
    if (enforcement_method_ == DirichletEnforcementMethod::DirectControl) {
      dU_dt_ = (U_ - u) / dt;
      U_     = u;
    }

    if (enforcement_method_ == DirichletEnforcementMethod::RateControl) {
      dU_dt_ = (U_plus_ - U_minus_) / (2.0 * epsilon);
      U_     = u;
    }

    if (enforcement_method_ == DirichletEnforcementMethod::FullControl) {
      dU_dt_ = (U_plus_ - U_minus_) / (2.0 * epsilon);
      U_     = U_ - dt * dU_dt_;
    }
  } else {
    dU_dt_ = (U_plus_ - U_minus_) / (2.0 * epsilon);
  }

  auto constrained_dofs = bcs_.allEssentialDofs();
  state_.u.SetSubVector(constrained_dofs, 0.0);
  U_.SetSubVectorComplement(constrained_dofs, 0.0);
  state_.u += U_;

  du_dt = state_.du_dt;
  du_dt.SetSubVector(constrained_dofs, 0.0);
  dU_dt_.SetSubVectorComplement(constrained_dofs, 0.0);
  du_dt += dU_dt_;

  solver_.Mult(zero_, du_dt);
  SLIC_WARNING_IF(!solver_.NonlinearSolver().GetConverged(), "Newton Solver did not converge.");

  state_.du_dt       = du_dt;
  state_.previous_dt = dt;
}

}  // namespace serac::mfem_ext

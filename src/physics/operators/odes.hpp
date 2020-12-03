// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <functional>

#include "mfem.hpp"
#include "numerics/expr_template_ops.hpp"
#include "physics/utilities/boundary_condition_manager.hpp"
#include "physics/utilities/equation_solver.hpp"

namespace serac {

/**
 * @brief SecondOrderODE is a class wrapping mfem::SecondOrderTimeDependentOperator
 *   so that the user can use std::function to define the implementations of
 *   mfem::SecondOrderTimeDependentOperator::Mult and
 *   mfem::SecondOrderTimeDependentOperator::ImplicitSolve
 *
 * The main benefit of this approach is that lambda capture lists allow
 * for a flexible inline representation of the overloaded functions,
 * without having to manually define a separate functor class.
 */
class SecondOrderODE : public mfem::SecondOrderTimeDependentOperator {
public:
  struct State {
    /**
     * @brief Current time step
     */
    double& c0;

    /**
     * @brief Previous value of dt
     */
    double& c1;

    /**
     * @brief Predicted true DOFs
     */
    mfem::Vector& u;

    /**
     * @brief  Predicted du_dt
     */
    mfem::Vector& du_dt;

    /**
     * @brief Previous value of d^2u_dt^2
     */
    mfem::Vector& d2u_dt2;
  };

  /**
   * @brief Constructor defining the size and specific system of ordinary differential equations to be solved
   *
   * @param[in] n The number of components in each vector of the ODE
   * @param[in] f The function that describing how to solve for the second derivative, given the current state
   *    and its time derivative. The two functions
   *
   *      mfem::SecondOrderTimeDependentOperator::Mult and mfem::SecondOrderTimeDependentOperator::ImplicitSolve
   *      (described in more detail here:
   * https://mfem.github.io/doxygen/html/classmfem_1_1SecondOrderTimeDependentOperator.html)
   *
   *    are consolidated into a single std::function, where
   *
   *      mfem::SecondOrderTimeDependentOperator::Mult corresponds to the case where fac0, fac1 are both zero
   *      mfem::SecondOrderTimeDependentOperator::ImplicitSolve corresponds to the case where either of fac0, fac1 are
   * nonzero
   *
   */
  SecondOrderODE(int n, State&& state, const EquationSolver& solver, const BoundaryConditionManager& bcs)
      : mfem::SecondOrderTimeDependentOperator(n, 0.0), state_(std::move(state)), solver_(solver), bcs_(bcs), zero_(n)
  {
    zero_ = 0.0;
    U_minus_.SetSize(n);
    U_.SetSize(n);
    U_plus_.SetSize(n);
    dU_dt_.SetSize(n);
    d2U_dt2_.SetSize(n);
  }

  void Mult(const mfem::Vector& u, const mfem::Vector& du_dt, mfem::Vector& d2u_dt2) const
  {
    Solve(t, 0.0, 0.0, u, du_dt, d2u_dt2);
  }

  void ImplicitSolve(const double c0, const double c1, const mfem::Vector& u, const mfem::Vector& du_dt,
                     mfem::Vector& d2u_dt2)
  {
    Solve(t, c0, c1, u, du_dt, d2u_dt2);
  }

private:
  void Solve(const double t, const double c0, const double c1, const mfem::Vector& u, const mfem::Vector& du_dt,
             mfem::Vector& d2u_dt2) const
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

    bool implicit = (c0 != 0.0 || c0 != 0.0);
    if (implicit) {
      if (enforcement_method_ == DirichletEnforcementMethod::DirectControl) {
        d2U_dt2_ = (U_ - u) / c0;
        dU_dt_   = du_dt;
        U_       = u;
      }

      if (enforcement_method_ == DirichletEnforcementMethod::RateControl) {
        d2U_dt2_ = (dU_dt_ - du_dt) / c0;
        dU_dt_   = du_dt;
        U_       = u;
      }

      if (enforcement_method_ == DirichletEnforcementMethod::FullControl) {
        d2U_dt2_ = (U_minus_ - 2.0 * U_ + U_plus_) / (epsilon * epsilon);
        dU_dt_   = (U_plus_ - U_minus_) / (2.0 * epsilon) - c0 * d2U_dt2_;
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
    SLIC_WARNING_IF(!solver_.nonlinearSolver().GetConverged(), "Newton Solver did not converge.");

    state_.d2u_dt2 = d2u_dt2;
  }
  State                           state_;
  DirichletEnforcementMethod      enforcement_method_;
  const EquationSolver&           solver_;
  const BoundaryConditionManager& bcs_;
  mfem::Vector                    zero_;
  mutable mfem::Vector            U_minus_;
  mutable mfem::Vector            U_;
  mutable mfem::Vector            U_plus_;
  mutable mfem::Vector            dU_dt_;
  mutable mfem::Vector            d2U_dt2_;
};

/**
 * @brief FirstOrderODE is a class wrapping mfem::TimeDependentOperator
 *   so that the user can use std::function to define the implementations of
 *   mfem::TimeDependentOperator::Mult and
 *   mfem::TimeDependentOperator::ImplicitSolve
 *
 * The main benefit of this approach is that lambda capture lists allow
 * for a flexible inline representation of the overloaded functions,
 * without having to manually define a separate functor class.
 */
class FirstOrderODE : public mfem::TimeDependentOperator {
public:
  struct State {
    /**
     * @brief Predicted true DOFs
     */
    mfem::Vector& u;

    /**
     * @brief Current time step
     */
    double& dt;

    /**
     * @brief Previous value of du_dt
     */
    mfem::Vector& du_dt;

    /**
     * @brief Previous value of dt
     */
    double& previous_dt;
  };

  /**
   * @brief Constructor defining the size and specific system of ordinary differential equations to be solved
   *
   * @param[in] n The number of components in each vector of the ODE
   * @param[in] f The function that describing how to solve for the first derivative, given the current state.
   *    The two functions
   *
   *      mfem::TimeDependentOperator::Mult and mfem::TimeDependentOperator::ImplicitSolve
   *      (described in more detail here: https://mfem.github.io/doxygen/html/classmfem_1_1TimeDependentOperator.html)
   *
   *    are consolidated into a single std::function, where
   *
   *      mfem::TimeDependentOperator::Mult corresponds to the case where dt is zero
   *      mfem::TimeDependentOperator::ImplicitSolve corresponds to the case where dt is nonzero
   *
   */
  FirstOrderODE(int n, State&& state, const EquationSolver& solver, const BoundaryConditionManager& bcs)
      : mfem::TimeDependentOperator(n, 0.0), state_(std::move(state)), solver_(solver), bcs_(bcs), zero_(n)
  {
    zero_ = 0.0;
    U_minus_.SetSize(n);
    U_.SetSize(n);
    U_plus_.SetSize(n);
    dU_dt_.SetSize(n);
  }

  void Mult(const mfem::Vector& u, mfem::Vector& du_dt) const { Solve(0.0, u, du_dt); }
  void ImplicitSolve(const double dt, const mfem::Vector& u, mfem::Vector& du_dt) { Solve(dt, u, du_dt); }

  void setEnforcementMethod(const DirichletEnforcementMethod method) { enforcement_method_ = method; }

  /**
   * @brief the function that is used to implement mfem::TDO::Mult and mfem::TDO::ImplicitSolve
   */

private:
  void Solve(const double dt, const mfem::Vector& u, mfem::Vector& du_dt) const
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
    SLIC_WARNING_IF(!solver_.nonlinearSolver().GetConverged(), "Newton Solver did not converge.");

    state_.du_dt       = du_dt;
    state_.previous_dt = dt;
  }

  State                           state_;
  DirichletEnforcementMethod      enforcement_method_;
  const EquationSolver&           solver_;
  const BoundaryConditionManager& bcs_;
  mfem::Vector                    zero_;
  mutable mfem::Vector            U_minus_;
  mutable mfem::Vector            U_;
  mutable mfem::Vector            U_plus_;
  mutable mfem::Vector            dU_dt_;
};

}  // namespace serac

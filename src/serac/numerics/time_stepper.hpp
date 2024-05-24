// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file time_stepper.hpp
 *
 * @brief Class which specifies how time or load steps are advanced, and their sensitivities
 */

#pragma once

#include <vector>
#include "serac/numerics/solver_config.hpp"
#include "mfem.hpp"  // MRT, work on removing this dependence

namespace serac {

class FiniteElementState;
class EquationSolver;
class BoundaryConditionManager;
struct TimesteppingOptions;
namespace mfem_ext {
class SecondOrderODE;
}

class TimeStepper {
public:
  virtual ~TimeStepper() {}
  virtual void set_states(const std::vector<FiniteElementState*>& states, BoundaryConditionManager& bcs) = 0;
  virtual void reset()                                                                                   = 0;
  virtual void advance(double t, double dt)                                                              = 0;
  virtual void reverse_vjp(double t, double dt)                                                          = 0;
};

class SecondOrderTimeStepper : public TimeStepper {
public:
  SecondOrderTimeStepper(EquationSolver* solver, const TimesteppingOptions& timestepping_opts);

  void set_states(const std::vector<FiniteElementState*>& states, BoundaryConditionManager& bcs) override;
  void reset() override;
  void advance(double t, double dt) override;
  void reverse_vjp(double t, double dt) override;

protected:
  /// The value of time at which the ODE solver wants to evaluate the residual
  double ode_time_point;

  /// coefficient used to calculate predicted displacement: u_p := u + c0 * d2u_dt2
  double c0;

  /// coefficient used to calculate predicted velocity: dudt_p := dudt + c1 * d2u_dt2
  double c1;

  /// @brief used to communicate the ODE solver's predicted displacement to the residual operator
  mfem::Vector u;

  /// @brief used to communicate the ODE solver's predicted velocity to the residual operator
  mfem::Vector v;

  /// @brief used to get the ODE solver's acceleration from the residual operator
  mfem::Vector a;

  /// the specific methods and tolerances specified to solve the nonlinear residual equations
  EquationSolver* nonlinear_solver;

  /// The timestepping options for the solid mechanics time evolution operator
  const TimesteppingOptions timestepping_options;

  /**
   * @brief the ordinary differential equation that describes
   * how to solve for the second time derivative of displacement, given
   * the current displacement, velocity, and source terms
   */
  std::unique_ptr<mfem_ext::SecondOrderODE> ode2;
};

class QuasiStaticStepper : public TimeStepper {
public:
  QuasiStaticStepper(EquationSolver* solver, const TimesteppingOptions& timestepping_opts);

  void set_states(const std::vector<FiniteElementState*>& states, BoundaryConditionManager& bcs) override;
  void reset() override;
  void advance(double t, double dt) override;
  void reverse_vjp(double t, double dt) override;

protected:
  /// The value of time at which the ODE solver wants to evaluate the residual
  double ode_time_point;

  /// coefficient used to calculate predicted displacement: u_p := u + c0 * d2u_dt2
  double c0;

  /// coefficient used to calculate predicted velocity: dudt_p := dudt + c1 * d2u_dt2
  double c1;

  /// @brief used to communicate the ODE solver's predicted displacement to the residual operator
  mfem::Vector u;

  /// @brief used to communicate the ODE solver's predicted velocity to the residual operator
  mfem::Vector v;

  /// @brief used to get the ODE solver's acceleration from the residual operator
  mfem::Vector a;

  /// the specific methods and tolerances specified to solve the nonlinear residual equations
  std::unique_ptr<EquationSolver> nonlinear_solver;

  /// The timestepping options for the solid mechanics time evolution operator
  const TimesteppingOptions timestepping_options;
};

}  // namespace serac
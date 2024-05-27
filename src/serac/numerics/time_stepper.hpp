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

  using FieldVec = std::vector<FiniteElementState*>;
  using VectorVec = std::vector<mfem::Vector*>;
  using ConstVectorVec = std::vector<const mfem::Vector*>;

  virtual void setStates(const FieldVec& independentStates,
                         const FieldVec& states,
                         BoundaryConditionManager& bcs) = 0;

  virtual void reset()                      = 0;
  virtual void step(double t, double dt)    = 0;
  virtual void vjpStep(double t, double dt) = 0;

  using ResidualFuncType = std::function<void(double time,
                                              const VectorVec& independentStates,
                                              const ConstVectorVec& states,
                                              mfem::Vector& residual)>;

  using JacobianFuncType = std::function<void(double time,
                                              const VectorVec& independentStates,
                                              const ConstVectorVec& states,
                                              std::unique_ptr<mfem::HypreParMatrix>& J)>;

  void setResidualFunc(const ResidualFuncType& f) {
    residual_func = f;
  }

  void setJacobianFunc(const JacobianFuncType& f) {
    jacobian_func = f;
  }

  void setFuncs() {
    //nonlin_solver_->setOperator(*residual_with_bcs_);
  }

  private:

  ResidualFuncType residual_func;
  JacobianFuncType jacobian_func;
};

class SecondOrderTimeStepper : public TimeStepper {
public:
  SecondOrderTimeStepper(EquationSolver* solver, const TimesteppingOptions& timestepping_opts);

  void setStates(const FieldVec& independentStates,
                 const FieldVec& states, BoundaryConditionManager& bcs) override;
  void reset() override;
  void step(double t, double dt) override;
  void vjpStep(double t, double dt) override;

protected:
  /// The value of time at which the ODE solver wants to evaluate the residual
  double ode_time_point_;

  /// coefficient used to calculate predicted displacement: u_p := u + c0 * d2u_dt2
  double c0_;

  /// coefficient used to calculate predicted velocity: dudt_p := dudt + c1 * d2u_dt2
  double c1_;

  /// @brief used to communicate the ODE solver's predicted displacement to the residual operator
  mfem::Vector u_;

  /// @brief used to communicate the ODE solver's predicted velocity to the residual operator
  mfem::Vector v_;

  /// the specific methods and tolerances specified to solve the nonlinear residual equations
  EquationSolver* nonlinear_solver_;

  /// The timestepping options for the solid mechanics time evolution operator
  const TimesteppingOptions timestepping_options_;

  /**
   * @brief the ordinary differential equation that describes
   * how to solve for the second time derivative of displacement, given
   * the current displacement, velocity, and source terms
   */
  std::unique_ptr<mfem_ext::SecondOrderODE> ode2_;

  FieldVec independentStates_;
  FieldVec states_;
};

class QuasiStaticStepper : public TimeStepper {
public:
  QuasiStaticStepper(EquationSolver* solver, const TimesteppingOptions& timestepping_opts);

  void setStates(const FieldVec& independentStates,
                 const FieldVec& states, BoundaryConditionManager& bcs) override;
  void reset() override;
  void step(double t, double dt) override;
  void vjpStep(double t, double dt) override;

protected:
  /// The value of time at which the ODE solver wants to evaluate the residual
  double ode_time_point_;

  /// @brief used to communicate the ODE solver's predicted displacement to the residual operator
  mfem::Vector u_;

  /// the specific methods and tolerances specified to solve the nonlinear residual equations
  std::unique_ptr<EquationSolver> nonlinear_solver_;
};

}  // namespace serac
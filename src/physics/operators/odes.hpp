#pragma once

#include <functional>
#include <variant>

#include "mfem.hpp"
#include "numerics/expr_template_ops.hpp"
#include "physics/utilities/boundary_condition_manager.hpp"
#include "physics/utilities/equation_solver.hpp"

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
  using TypeSignature = void(const double, const double, const double, const mfem::Vector&, const mfem::Vector&, mfem::Vector&);

  /**
   * @brief Default constructor for creating an uninitialized SecondOrderODE
   */
  SecondOrderODE() : mfem::SecondOrderTimeDependentOperator(0, 0.0) {}

  /**
   * @brief Constructor defining the size and specific system of ordinary differential equations to be solved
   * 
   * @param[in] n The number of components in each vector of the ODE
   * @param[in] f The function that describing how to solve for the second derivative, given the current state 
   *    its time derivative. The two functions 
   * 
   *      mfem::SecondOrderTimeDependentOperator::Mult and mfem::SecondOrderTimeDependentOperator::ImplicitSolve
   *      (described in more detail here: https://mfem.github.io/doxygen/html/classmfem_1_1SecondOrderTimeDependentOperator.html)
   * 
   *    are consolidated into a single std::function, where 
   * 
   *      mfem::SecondOrderTimeDependentOperator::Mult corresponds to the case where fac0, fac1 are both zero
   *      mfem::SecondOrderTimeDependentOperator::Mult corresponds to the case where either of fac0, fac1 are nonzero
   * 
   */
  SecondOrderODE(int n, std::function<TypeSignature> f) : mfem::SecondOrderTimeDependentOperator(n, 0.0), f_(f) {}

  void Mult(const mfem::Vector& u, const mfem::Vector& du_dt, mfem::Vector& d2u_dt2) const
  {
    f_(t, 0.0, 0.0, u, du_dt, d2u_dt2);
  }

  void ImplicitSolve(const double c0, const double c1, const mfem::Vector& u, const mfem::Vector& du_dt,
                     mfem::Vector& d2u_dt2)
  {
    f_(t, c0, c1, u, du_dt, d2u_dt2);
  }

 private:

  /**
   * @brief the function that is used to implement mfem::SOTDO::Mult and mfem::SOTDO::ImplicitSolve
   */
  std::function<TypeSignature> f_;
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
  using TypeSignature = void(const double, const double, const mfem::Vector&, mfem::Vector&);

  /**
   * @brief Default constructor for creating an uninitialized FirstOrderODE
   */
  FirstOrderODE() : mfem::TimeDependentOperator(0, 0.0) {}

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
   *      mfem::TimeDependentOperator::Mult corresponds to the case where dt is nonzero
   * 
   */
  FirstOrderODE(int n, std::function<TypeSignature> f) : mfem::TimeDependentOperator(n, 0.0), f_(f) {}

  void Mult(const mfem::Vector& u, mfem::Vector& du_dt) const { f_(t, 0.0, u, du_dt); }
  void ImplicitSolve(const double dt, const mfem::Vector& u, mfem::Vector& du_dt) { f_(t, dt, u, du_dt); }

  /**
   * @brief the function that is used to implement mfem::TDO::Mult and mfem::TDO::ImplicitSolve
   */

 private:
  std::function<TypeSignature> f_;
};

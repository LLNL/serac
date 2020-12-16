// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <functional>

#include "mfem.hpp"

/**
 * @brief StdFunctionOperator is a class wrapping mfem::Operator
 *   so that the user can use std::function to define the implementations of
 *   mfem::Operator::Mult and
 *   mfem::Operator::GetGradient
 *
 * The main benefit of this approach is that lambda capture lists allow
 * for a flexible inline representation of the overloaded functions,
 * without having to manually define a separate functor class.
 */
class StdFunctionOperator : public mfem::Operator {
public:
  /**
   * @brief Default constructor for creating an uninitialized StdFunctionOperator
   */
  StdFunctionOperator(int n) : mfem::Operator(n) {}

  /**
   * @brief Constructor for a StdFunctionOperator that only defines mfem::Operator::Mult
   */
  StdFunctionOperator(int n, std::function<void(const mfem::Vector&, mfem::Vector&)> function)
      : mfem::Operator(n), function_(function)
  {
  }

  /**
   * @brief Constructor for a StdFunctionOperator that defines mfem::Operator::Mult and mfem::Operator::GetGradient
   */
  StdFunctionOperator(int n, std::function<void(const mfem::Vector&, mfem::Vector&)> function,
                      std::function<mfem::Operator&(const mfem::Vector&)> jacobian)
      : mfem::Operator(n), function_(function), jacobian_(jacobian)
  {
  }

  void            Mult(const mfem::Vector& k, mfem::Vector& y) const { function_(k, y); };
  mfem::Operator& GetGradient(const mfem::Vector& k) const { return jacobian_(k); };

private:
  /**
   * @brief the function that is used to implement mfem::Operator::Mult
   */
  std::function<void(const mfem::Vector&, mfem::Vector&)> function_;

  /**
   * @brief the function that is used to implement mfem::Operator::GetGradient
   */
  std::function<mfem::Operator&(const mfem::Vector&)> jacobian_;
};

// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file vector_expression.hpp
 *
 * @brief A set of template classes used to represent the evaluation of unary and binary operations
 * on vectors
 */

#pragma once

#include "mfem.hpp"

namespace serac {
/**
 * @brief A base class representing a vector expression
 * @tparam T The base vector type, e.g., mfem::Vector, or another VectorExpr
 * @note This class should never be used directly
 */
template <typename T>
class VectorExpr {
public:
  /**
   * @brief Returns the fully evaluated value for the vector
   * expression at index @p i
   * @param i The index to evaluate at
   */
  double operator[](int i) const { return asDerived()[i]; }

  /**
   * @brief Returns the size of the vector expression
   */
  int Size() const { return asDerived().Size(); }

  /**
   * @brief Implicit conversion operator for fully evaluating
   * a vector expression into an actual mfem::Vector
   * @return The fully evaluated vector
   */
  operator mfem::Vector() const
  {
    mfem::Vector result(Size());
    for (int i = 0; i < Size(); i++) {
      result[i] = (*this)[i];
    }
    return result;
  }

  /**
   * @brief Performs a compile-time downcast to the derived object
   * @return The derived object
   * @see Curiously Recurring Template Pattern
   */
  const T& asDerived() const { return static_cast<const T&>(*this); }

  /**
   * @brief Performs a compile-time downcast to the derived object
   * @return The derived object
   * @see Curiously Recurring Template Pattern
   */
  T& asDerived() { return static_cast<T&>(*this); }
};

/**
 * @brief Fully evaluates a vector expression into an actual mfem::Vector
 * @param expr The expression to evaluate
 * @return The fully evaluated vector
 * @see VectorExpr::operator mfem::Vector
 */
template <typename T>
mfem::Vector evaluate(const VectorExpr<T>& expr)
{
  return expr;
}

/**
 * @brief Fully evaluates a vector expression into an actual mfem::Vector
 * @param expr The expression to evaluate
 * @param result The vector to populate with the expression result
 */
template <typename T>
void evaluate(const VectorExpr<T>& expr, mfem::Vector& result)
{
  SLIC_ERROR_IF(expr.Size() != result.Size(), "Vector sizes in expression assignment must be equal");
  // Get the underlying array for indexing compatibility with mfem::HypreParVector
  double* result_arr = result;
  for (int i = 0; i < expr.Size(); i++) {
    result_arr[i] = expr[i];
  }
}

}  // namespace serac

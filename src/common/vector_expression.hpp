// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
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

#ifndef VECTOR_EXPRESSION
#define VECTOR_EXPRESSION

#include "mfem.hpp"

namespace serac {
/**
 * @brief A base class representing a vector expression
 * @tparam T The base vector type, e.g., mfem::Vector, or another VectorExpr
 * @note This class should never be used directly
 */
template <typename T>
class VectorExpr {
#ifndef NDEBUG
#warning The use of expression templates in debug builds has a significant performance cost (up to 20x)
#endif
public:
  /**
   * @brief Returns the fully evaluated value for the vector
   * expression at index @p i
   * @param i The index to evaluate at
   */
  double operator[](size_t i) const { return asDerived()[i]; }

  /**
   * @brief Returns the size of the vector expression
   */
  size_t Size() const { return asDerived().Size(); }

  /**
   * @brief Implicit conversion operator for fully evaluating
   * a vector expression into an actual mfem::Vector
   * @return The fully evaluated vector
   */
  operator mfem::Vector() const
  {
    mfem::Vector result(Size());
    for (size_t i = 0; i < Size(); i++) {
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
 * @return The fully evaluated vector
 * @see VectorExpr::operator mfem::Vector
 */
template <typename T>
mfem::Vector evaluate(const VectorExpr<T>& expr)
{
  return expr;
}

}  // namespace serac

#endif

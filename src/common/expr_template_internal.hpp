// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * \@file expr_templates_internal.hpp
 *
 * @brief The internal implementation for a set of template classes used to
 * represent the evaluation of unary and binary operations on vectors
 */

#ifndef EXPR_TEMPLATES_INTERNAL
#define EXPR_TEMPLATES_INTERNAL

#include <functional>
#include <type_traits>
#include <utility>

#include "common/logger.hpp"
#include "common/vector_expression.hpp"

namespace serac::internal {

/**
 * @brief Determines whether a given type should be owned by a vector expression
 * @tparam vec The vector expression type
 * @note A vector should be owned if it is not an (optionally const) reference
 * to an mfem::Vector.  Note that std::remove_const does not apply here because
 * const is not a top-level qualifier on a const reference
 */
template <typename vec>
inline constexpr bool owns_v = !std::is_same_v<vec, const mfem::Vector&> && !std::is_same_v<vec, mfem::Vector&>;

/**
 * @brief Type alias for what should be stored by a vector expression - an object
 * if ownership is desired, otherwise, a reference
 * @tparam vec The base vector expression type
 */
template <typename vec>
using vec_t = typename std::conditional_t<owns_v<vec>, std::decay_t<vec>, const std::decay_t<vec>&>;

/**
 * @brief Type alias for the constructor argument to a vector expression - an
 * rvalue reference if ownership is desired (will be moved from), otherwise,
 * an lvalue reference
 * @tparam vec The base vector expression type
 */
template <typename vec>
using vec_arg_t = typename std::conditional_t<owns_v<vec>, std::decay_t<vec>&&, const vec&>;

/**
 * @brief Derived VectorExpr class for representing the application of a unary
 * operator to a vector
 * @tparam vec The base vector type, e.g., mfem::Vector, or another VectorExpr
 * @tparam UnOp The type of the unary operator
 * @pre UnOp must be a functor with the following signature:
 * @code{.cpp}
 * double UnOp::operator()(const double arg);
 * @endcode
 */
template <typename vec, typename UnOp>
class UnaryVectorExpr : public VectorExpr<UnaryVectorExpr<vec, UnOp>> {
public:
  /**
   * @brief Constructs an element-wise unary expression on a vector
   */
  UnaryVectorExpr(vec_arg_t<vec> v, UnOp&& op = UnOp{}) : v_(std::forward<vec_t<vec>>(v)), op_(std::move(op)) {}
  /**
   * @brief Returns the fully evaluated value for the vector
   * expression at index @p i
   * @param i The index to evaluate at
   */
  double operator[](size_t i) const { return op_(v_[i]); }
  /**
   * @brief Returns the size of the vector expression
   */
  size_t Size() const { return v_.Size(); }

private:
  const vec_t<vec> v_;
  const UnOp       op_;
};

/**
 * @brief Functor class for binding a scalar to a multiplication operation
 */
class ScalarMultOp {
public:
  /**
   * @brief Constructs a partial application of a scalar multiplication
   */
  ScalarMultOp(const double scalar) : scalar_(scalar) {}

  /**
   * @brief Applies the partial application to the remaining argument
   */
  double operator()(const double arg) const { return arg * scalar_; }

private:
  double scalar_;
};

/**
 * @brief Functor class for binding a scalar to a division operation
 * @tparam is_denominator Whether the scalar is the divisor/denominator (vec divided by scalar)
 * or the dividend (scalar divided by vector)
 */
template <bool is_denominator = true>
class ScalarDivOp {
public:
  /**
   * @brief Constructs a partial application of a scalar division
   */
  ScalarDivOp(const double scalar) : scalar_(scalar) {}

  /**
   * @brief Applies the partial application to the remaining argument
   */
  double operator()(const double arg) const
  {
    if constexpr (is_denominator) {
      return arg / scalar_;
    } else {
      return scalar_ / arg;
    }
  }

private:
  double scalar_;
};

/**
 * @brief Type alias for a vector negation operation
 */
template <typename vec>
using UnaryNegation = UnaryVectorExpr<vec, std::negate<double>>;

/**
 * @brief Derived VectorExpr class for representing the application of a binary
 * operator to two vectors
 * @tparam lhs The base vector type for the expression LHS, e.g., mfem::Vector,
 * or another VectorExpr
 * @tparam rhs The base vector type for the expression RHS, e.g., mfem::Vector,
 * or another VectorExpr
 * @tparam BinOp The type of the binary operator
 * @pre UnOp must be a functor with the following signature:
 * @code{.cpp}
 * double BinOp::operator()(const double lhs, const double rhs);
 * @endcode
 */
template <typename lhs, typename rhs, typename BinOp>
class BinaryVectorExpr : public VectorExpr<BinaryVectorExpr<lhs, rhs, BinOp>> {
public:
  /**
   * @brief Constructs an element-wise binary expression on two vectors
   */
  BinaryVectorExpr(vec_arg_t<lhs> u, vec_arg_t<rhs> v)
      : u_(std::forward<vec_t<lhs>>(u)), v_(std::forward<vec_t<rhs>>(v))
  {
    // MFEM uses int to represent a size type, so cast to size_t for consistency
    SLIC_ERROR_IF(static_cast<std::size_t>(u_.Size()) != static_cast<std::size_t>(v_.Size()),
                  "Vector sizes in binary operation must be equal");
  }
  /**
   * @brief Returns the fully evaluated value for the vector
   * expression at index @p i
   * @param i The index to evaluate at
   */
  double operator[](size_t i) const { return op_(u_[i], v_[i]); }
  /**
   * @brief Returns the size of the vector expression
   */
  size_t Size() const { return v_.Size(); }

private:
  const vec_t<lhs> u_;
  const vec_t<rhs> v_;
  const BinOp      op_ = BinOp{};
};

/**
 * @brief Type alias for a vector addition operation
 */
template <typename lhs, typename rhs>
using VectorAddition = BinaryVectorExpr<lhs, rhs, std::plus<double>>;

/**
 * @brief Type alias for a vector subtraction operation
 */
template <typename lhs, typename rhs>
using VectorSubtraction = BinaryVectorExpr<lhs, rhs, std::minus<double>>;

/**
 * @brief Derived VectorExpr class for the application of an mfem::Operator to a vector,
 * e.g., matrix-vector multiplication
 * @tparam vec The base vector type, e.g., mfem::Vector, or another VectorExpr
 * @pre The mfem::Operator must have its `height` member variable set to a
 * nonzero value
 * @note This class does not participate in lazy evaluation, that is, it
 * will perform the full operation (`mfem::Operator::Mult`) when the object
 * is constructed
 */
template <typename vec>
class OperatorExpr : public VectorExpr<OperatorExpr<vec>> {
public:
  /**
   * @brief Constructs a "mfem::Operator::Mult" expression
   */
  OperatorExpr(const mfem::Operator& A, const vec& v) : result_(A.Height())
  {
    // No-op if `vec` is a already an mfem::Vector, otherwise explicitly convert
    A.Mult(static_cast<mfem::Vector>(v), result_);
  }
  /**
   * @brief Returns the fully evaluated value for the vector
   * expression at index @p i
   * @param i The index to evaluate at
   */
  double operator[](size_t i) const { return result_[i]; }
  /**
   * @brief Returns the size of the vector expression
   */
  size_t Size() const { return result_.Size(); }

private:
  mfem::Vector result_;
};

}  // namespace serac::internal

#endif

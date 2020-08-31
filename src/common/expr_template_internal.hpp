// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file expr_templates_internal.hpp
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

// Type alias for what should be stored by a vector expression - an object
// if ownership is desired, otherwise, a reference
template <typename vec, bool owns>
using vec_t = typename std::conditional<owns, vec, const vec&>::type;

// Type alias for the constructior argument to a vector expression - an
// rvalue reference if ownership is desired (will be moved from), otherwise,
// and lvalue reference
template <typename vec, bool owns>
using vec_arg_t = typename std::conditional<owns, std::remove_const_t<vec>&&, const vec&>::type;

/**
 * @brief Derived VectorExpr class for representing the application of a unary
 * operator to a vector
 * @tparam vec The base vector type, e.g., mfem::Vector, or another VectorExpr
 * @tparam owns Whether the object owns its vector
 * @tparam UnOp The type of the unary operator
 * @pre UnOp must be a functor with the following signature:
 * @code{.cpp}
 * double UnOp::operator()(const double arg);
 * @endcode
 */
template <typename vec, bool owns, typename UnOp>
class UnaryVectorExpr : public VectorExpr<UnaryVectorExpr<vec, owns, UnOp>> {
public:
  UnaryVectorExpr(vec_arg_t<vec, owns> v, UnOp&& op = UnOp{})
      : v_(std::forward<vec_t<vec, owns>>(v)), op_(std::move(op))
  {
  }
  double operator[](size_t i) const { return op_(v_[i]); }
  size_t Size() const { return v_.Size(); }

private:
  const vec_t<vec, owns> v_;
  const UnOp             op_;
};

/**
 * @brief Functor class for binding a scalar to a multiplication operatior
 */
class ScalarMultOp {
public:
  ScalarMultOp(const double scalar) : scalar_(scalar) {}
  double operator()(const double arg) const { return arg * scalar_; }

private:
  double scalar_;
};

template <typename vec, bool owns>
using UnaryNegation = UnaryVectorExpr<vec, owns, std::negate<double>>;

/**
 * @brief Derived VectorExpr class for representing the application of a binary
 * operator to two vectors
 * @tparam lhs The base vector type for the expression LHS, e.g., mfem::Vector,
 * or another VectorExpr
 * @tparam rhs The base vector type for the expression RHS, e.g., mfem::Vector,
 * or another VectorExpr
 * @tparam lhs_owns Whether the object owns its LHS vector
 * @tparam rhs_owns Whether the object owns its RHS vector
 * @tparam BinOp The type of the binary operator
 * @pre UnOp must be a functor with the following signature:
 * @code{.cpp}
 * double BinOp::operator()(const double lhs, const double rhs);
 * @endcode
 */
template <typename lhs, typename rhs, bool lhs_owns, bool rhs_owns, typename BinOp>
class BinaryVectorExpr : public VectorExpr<BinaryVectorExpr<lhs, rhs, lhs_owns, rhs_owns, BinOp>> {
public:
  BinaryVectorExpr(vec_arg_t<lhs, lhs_owns> u, vec_arg_t<rhs, rhs_owns> v)
      : u_(std::forward<vec_t<lhs, lhs_owns>>(u)), v_(std::forward<vec_t<rhs, rhs_owns>>(v))
  {
    // MFEM uses int to represent a size typw, so cast to size_t for consistency
    SLIC_ERROR_IF(static_cast<std::size_t>(u_.Size()) != static_cast<std::size_t>(v_.Size()),
                  "Vector sizes in binary operation must be equal");
  }
  double operator[](size_t i) const { return op_(u_[i], v_[i]); }
  size_t Size() const { return v_.Size(); }

private:
  const vec_t<lhs, lhs_owns> u_;
  const vec_t<rhs, rhs_owns> v_;
  const BinOp                op_ = BinOp{};
};

template <typename lhs, typename rhs, bool lhs_owns, bool rhs_owns>
using VectorAddition = BinaryVectorExpr<lhs, rhs, lhs_owns, rhs_owns, std::plus<double>>;

template <typename lhs, typename rhs, bool lhs_owns, bool rhs_owns>
using VectorSubtraction = BinaryVectorExpr<lhs, rhs, lhs_owns, rhs_owns, std::minus<double>>;

/**
 * @brief Derived VectorExpr class for the application of an mfem::Operator to a vector,
 * e.g., matrix-vector multiplication
 * @tparam vec The base vector type, e.g., mfem::Vector, or another VectorExpr
 * @tparam owns Whether the object owns its vector
 * @pre The mfem::Operator must have its `height` member variable set to a
 * nonzero value
 * @note This class does not participate in lazy evaluation, that is, it
 * will perform the full operation (`mfem::Operator::Mult`) when the object
 * is constructed
 */
template <typename vec, bool owns>
class OperatorExpr : public VectorExpr<OperatorExpr<vec, owns>> {
public:
  OperatorExpr(const mfem::Operator& A, vec_arg_t<vec, owns> v)
      : A_(A), v_(std::forward<vec_t<vec, owns>>(v)), result_(A_.Height())
  {
    if constexpr (std::is_same<vec, mfem::Vector>::value) {
      A_.Mult(v_, result_);
    } else {
      mfem::Vector tmp = evaluate(v_);
      A_.Mult(tmp, result_);
    }
  }
  double operator[](size_t i) const { return result_[i]; }
  size_t Size() const { return result_.Size(); }

private:
  const mfem::Operator&  A_;
  const vec_t<vec, owns> v_;
  mfem::Vector           result_;
};

}  // namespace serac::internal

#endif

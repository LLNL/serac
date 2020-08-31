// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef EXPR_TEMPLATES
#define EXPR_TEMPLATES

#include <functional>
#include <type_traits>
#include <utility>

#include "common/logger.hpp"
#include "mfem.hpp"

template <typename T>
class VectorExpr {
public:
  double operator[](size_t i) const { return asDerived()[i]; }
  size_t Size() const { return asDerived().Size(); }

  operator mfem::Vector() const
  {
    mfem::Vector result(Size());
    for (size_t i = 0; i < Size(); i++) {
      result[i] = (*this)[i];
    }
    return result;
  }
  const T& asDerived() const { return static_cast<const T&>(*this); }
  T&       asDerived() { return static_cast<T&>(*this); }
};

template <typename T>
mfem::Vector evaluate(const VectorExpr<T>& expr)
{
  return expr;
}

template <typename vec, bool owns>
using vec_t = typename std::conditional<owns, vec, const vec&>::type;

template <typename vec, bool owns>
using vec_arg_t = typename std::conditional<owns, std::remove_const_t<vec>&&, const vec&>::type;

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

class ScalarMultOp {
public:
  ScalarMultOp(const double scalar) : scalar_(scalar) {}
  double operator()(const double arg) const { return arg * scalar_; }

private:
  double scalar_;
};

template <typename vec, bool owns>
using UnaryNegation = UnaryVectorExpr<vec, owns, std::negate<double>>;

template <typename T>
auto operator-(VectorExpr<T>&& u)
{
  return UnaryNegation<T, true>(std::move(u.asDerived()));
}

inline auto operator-(const mfem::Vector& u) { return UnaryNegation<mfem::Vector, false>(u); }

template <typename T>
auto operator*(VectorExpr<T>&& u, const double a)
{
  return UnaryVectorExpr<T, true, ScalarMultOp>(std::move(u.asDerived()), ScalarMultOp{a});
}

template <typename T>
auto operator*(const double a, VectorExpr<T>&& u)
{
  return operator*(std::move(u), a);
}

inline auto operator*(const double a, const mfem::Vector& u)
{
  return UnaryVectorExpr<mfem::Vector, false, ScalarMultOp>(u, ScalarMultOp{a});
}

inline auto operator*(const mfem::Vector& u, const double a) { return operator*(a, u); }

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

template <typename S, typename T>
auto operator+(VectorExpr<S>&& u, VectorExpr<T>&& v)
{
  return VectorAddition<S, T, true, true>(std::move(u.asDerived()), std::move(v.asDerived()));
}

template <typename T>
auto operator+(const mfem::Vector& u, VectorExpr<T>&& v)
{
  return VectorAddition<mfem::Vector, T, false, true>(u, std::move(v.asDerived()));
}

template <typename T>
auto operator+(VectorExpr<T>&& u, const mfem::Vector& v)
{
  return operator+(v, std::move(u));
}

inline auto operator+(const mfem::Vector& u, const mfem::Vector& v)
{
  return VectorAddition<mfem::Vector, mfem::Vector, false, false>(u, v);
}

template <typename S, typename T>
auto operator-(VectorExpr<S>&& u, VectorExpr<T>&& v)
{
  return VectorSubtraction<S, T, true, true>(std::move(u.asDerived()), std::move(v.asDerived()));
}

template <typename T>
auto operator-(const mfem::Vector& u, VectorExpr<T>&& v)
{
  return VectorSubtraction<mfem::Vector, T, false, true>(u, std::move(v.asDerived()));
}

template <typename T>
auto operator-(VectorExpr<T>&& u, const mfem::Vector& v)
{
  return operator-(v, std::move(u));
}

inline auto operator-(const mfem::Vector& u, const mfem::Vector& v)
{
  return VectorSubtraction<mfem::Vector, mfem::Vector, false, false>(u, v);
}

template <typename vec, bool owns>
class Matvec : public VectorExpr<Matvec<vec, owns>> {
public:
  Matvec(const mfem::Operator& A, vec_arg_t<vec, owns> v)
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

template <typename T>
auto operator*(const mfem::Operator& A, VectorExpr<T>&& v)
{
  return Matvec<T, true>(A, std::move(v.asDerived()));
}

inline auto operator*(const mfem::Operator& A, const mfem::Vector& v) { return Matvec<mfem::Vector, false>(A, v); }

#endif

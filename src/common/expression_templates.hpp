#ifndef EXPR_TEMPLATES
#define EXPR_TEMPLATES

#include <assert.h>

#include <iostream>
#include <type_traits>
#include <utility>

#include "mfem.hpp"

template <typename T>
struct VectorExpr {
  double operator[](size_t i) const { return asDerived()[i]; }
  size_t Size() const { return asDerived().Size(); }

  const T& asDerived() const { return static_cast<const T&>(*this); }
  T&       asDerived() { return static_cast<T&>(*this); }
};

template <typename T>
auto evaluate(const VectorExpr<T>& expr)
{
  mfem::Vector result(expr.Size());
  for (size_t i = 0; i < expr.Size(); i++) {
    result[i] = expr[i];
  }
  return result;
}

template <typename vec, bool owns>
using vec_t = typename std::conditional<owns, vec, const vec&>::type;

template <typename vec, bool owns>
using vec_arg_t = typename std::conditional<owns, std::remove_const_t<vec>&&, const vec&>::type;

template <typename vec, bool owns>
struct UnaryNegation : VectorExpr<UnaryNegation<vec, owns>> {
  const vec_t<vec, owns> v_;
  UnaryNegation(vec_arg_t<vec, owns> v) : v_(std::forward<vec_t<vec, owns>>(v)) {}
  double operator[](size_t i) const { return -v_[i]; }
  size_t Size() const { return v_.Size(); }
};
template <typename T>
UnaryNegation(const T&) -> UnaryNegation<T, false>;

template <typename T>
UnaryNegation(T &&) -> UnaryNegation<T, true>;

template <typename T>
auto operator-(VectorExpr<T>&& u)
{
  return UnaryNegation(u.asDerived());
}

auto operator-(const mfem::Vector& u) { return UnaryNegation(u); }

template <typename vec, bool owns>
struct ScalarMultiplication : VectorExpr<ScalarMultiplication<vec, owns>> {
  const double           a_;
  const vec_t<vec, owns> v_;
  ScalarMultiplication(const double a, vec_arg_t<vec, owns> v) : a_(a), v_(std::forward<vec_t<vec, owns>>(v)) {}
  double operator[](size_t i) const { return a_ * v_[i]; }
  size_t Size() const { return v_.Size(); }
};
template <typename T>
ScalarMultiplication(const double, const T&) -> ScalarMultiplication<T, false>;

template <typename T>
ScalarMultiplication(const double, T &&) -> ScalarMultiplication<T, true>;

template <typename T>
auto operator*(VectorExpr<T>&& u, const double a)
{
  return ScalarMultiplication(a, std::move(u.asDerived()));
}

template <typename T>
auto operator*(const double a, VectorExpr<T>&& u)
{
  return ScalarMultiplication(a, std::move(u));
}

auto operator*(const double a, const mfem::Vector& u) { return ScalarMultiplication(a, u); }

auto operator*(const mfem::Vector& u, const double a) { return ScalarMultiplication(a, u); }

template <typename lhs, typename rhs, bool lhs_owns, bool rhs_owns>
struct VectorAddition : VectorExpr<VectorAddition<lhs, rhs, lhs_owns, rhs_owns>> {
  const vec_t<lhs, lhs_owns> u_;
  const vec_t<rhs, rhs_owns> v_;
  VectorAddition(vec_arg_t<lhs, lhs_owns> u, vec_arg_t<rhs, rhs_owns> v)
      : u_(std::forward<vec_t<lhs, lhs_owns>>(u)), v_(std::forward<vec_t<rhs, rhs_owns>>(v))
  {
    assert(u_.Size() == v_.Size());
  }
  double operator[](size_t i) const { return u_[i] + v_[i]; }
  size_t Size() const { return v_.Size(); }
};
template <typename S, typename T>
VectorAddition(const S&, const T&) -> VectorAddition<S, T, false, false>;

template <typename S, typename T>
VectorAddition(S&&, const T&) -> VectorAddition<S, T, true, false>;

template <typename S, typename T>
VectorAddition(const S&, T &&) -> VectorAddition<S, T, false, true>;

template <typename S, typename T>
VectorAddition(S&&, T &&) -> VectorAddition<S, T, true, true>;

template <typename S, typename T>
auto operator+(VectorExpr<S>&& u, VectorExpr<T>&& v)
{
  return VectorAddition(std::move(u.asDerived()), std::move(v.asDerived()));
}

template <typename T>
auto operator+(const mfem::Vector& u, VectorExpr<T>&& v)
{
  return VectorAddition(u, std::move(v.asDerived()));
}

template <typename T>
auto operator+(VectorExpr<T>&& u, const mfem::Vector& v)
{
  return VectorAddition(std::move(u.asDerived()), v);
}

auto operator+(const mfem::Vector& u, const mfem::Vector& v) { return VectorAddition(u, v); }

template <typename lhs, typename rhs, bool lhs_owns, bool rhs_owns>
struct VectorSubtraction : VectorExpr<VectorSubtraction<lhs, rhs, lhs_owns, rhs_owns>> {
  const vec_t<lhs, lhs_owns> u_;
  const vec_t<rhs, rhs_owns> v_;
  VectorSubtraction(vec_arg_t<lhs, lhs_owns> u, vec_arg_t<rhs, rhs_owns> v)
      : u_(std::forward<vec_t<lhs, lhs_owns>>(u)), v_(std::forward<vec_t<rhs, rhs_owns>>(v))
  {
    assert(u_.Size() == v_.Size());
  }
  double operator[](size_t i) const { return u_[i] - v_[i]; }
  size_t Size() const { return v_.Size(); }
};
template <typename S, typename T>
VectorSubtraction(const S&, const T&) -> VectorSubtraction<S, T, false, false>;

template <typename S, typename T>
VectorSubtraction(S&&, const T&) -> VectorSubtraction<S, T, true, false>;

template <typename S, typename T>
VectorSubtraction(const S&, T &&) -> VectorSubtraction<S, T, false, true>;

template <typename S, typename T>
VectorSubtraction(S&&, T &&) -> VectorSubtraction<S, T, true, true>;

template <typename S, typename T>
auto operator-(VectorExpr<S>&& u, VectorExpr<T>&& v)
{
  return VectorSubtraction(std::move(u.asDerived()), std::move(v.asDerived()));
}

template <typename T>
auto operator-(const mfem::Vector& u, VectorExpr<T>&& v)
{
  return VectorSubtraction(u, std::move(v.asDerived()));
}

template <typename T>
auto operator-(VectorExpr<T>&& u, const mfem::Vector& v)
{
  return VectorSubtraction(std::move(u.asDerived()), v);
}

auto operator-(const mfem::Vector& u, const mfem::Vector& v) { return VectorSubtraction(u, v); }

template <typename vec, bool owns>
struct Matvec : VectorExpr<Matvec<vec, owns>> {
  const mfem::Operator&  A_;
  const vec_t<vec, owns> v_;
  mfem::Vector           result;
  Matvec(const mfem::Operator& A, vec_arg_t<vec, owns> v) : A_(A), v_(std::forward<vec_t<vec, owns>>(v))
  {
    result.SetSize(A_.Height());
    if constexpr (std::is_same<vec, mfem::Vector>::value) {
      A_.Mult(v_, result);
    } else {
      mfem::Vector tmp = evaluate(v_);
      A_.Mult(tmp, result);
    }
  }
  double operator[](size_t i) const { return result[i]; }
  size_t Size() const { return result.Size(); }
};
template <typename T>
Matvec(const mfem::Operator& A, const T&) -> Matvec<T, false>;

template <typename T>
Matvec(const mfem::Operator& A, T &&) -> Matvec<T, true>;

template <typename T>
auto operator*(const mfem::Operator& A, VectorExpr<T>&& v)
{
  return Matvec(A, std::move(v.asDerived()));
}

auto operator*(const mfem::Operator& A, const mfem::Vector& v) { return Matvec(A, v); }

#endif

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

template <typename vec>
struct UnaryNegation : VectorExpr<UnaryNegation<vec> > {
  const vec& v_;
  UnaryNegation(const vec& v) : v_(v) {}
  double operator[](size_t i) const { return -v_[i]; }
  size_t Size() const { return v_.Size(); }

  ~UnaryNegation() { std::cout << "un dtor\n"; }
};
template <typename T>
UnaryNegation(const T&) -> UnaryNegation<T>;
template <typename T>
auto operator-(const VectorExpr<T>& u)
{
  return UnaryNegation(u.asDerived());
}

auto operator-(const mfem::Vector& u) { return UnaryNegation(u); }

template <typename vec>
struct ScalarMultiplication : VectorExpr<ScalarMultiplication<vec> > {
  const double a_;
  const vec    v_;
  ScalarMultiplication(const double a, vec&& v) : a_(a), v_(std::move(v)) {}
  double operator[](size_t i) const { return a_ * v_[i]; }
  size_t Size() const { return v_.Size(); }
};
template <typename T>
ScalarMultiplication(const double, const T&) -> ScalarMultiplication<T>;

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

auto operator*(const double a, mfem::Vector& u) { return ScalarMultiplication(a, std::move(u)); }

auto operator*(mfem::Vector& u, const double a) { return ScalarMultiplication(a, std::move(u)); }

template <typename lhs, typename rhs>
struct VectorAddition : VectorExpr<VectorAddition<lhs, rhs> > {
  const lhs& u;
  const rhs& v;
  VectorAddition(const lhs& u, const rhs& v) : u(u), v(v) { assert(u.Size() == v.Size()); }
  double operator[](size_t i) const { return u[i] + v[i]; }
  size_t Size() const { return v.Size(); }
};
template <typename S, typename T>
VectorAddition(const S&, const T&) -> VectorAddition<S, T>;

template <typename S, typename T>
auto operator+(const VectorExpr<S>& u, const VectorExpr<T>& v)
{
  return VectorAddition(u.asDerived(), v.asDerived());
}

template <typename T>
auto operator+(const mfem::Vector& u, const VectorExpr<T>& v)
{
  return VectorAddition(u, v.asDerived());
}

template <typename T>
auto operator+(const VectorExpr<T>& u, const mfem::Vector& v)
{
  return VectorAddition(u.asDerived(), v);
}

auto operator+(const mfem::Vector& u, const mfem::Vector& v) { return VectorAddition(u, v); }

template <typename lhs, typename rhs>
struct VectorSubtraction : VectorExpr<VectorSubtraction<lhs, rhs> > {
  const lhs& u;
  const rhs& v;
  VectorSubtraction(const lhs& u, const rhs& v) : u(u), v(v) { assert(u.Size() == v.Size()); }
  double operator[](size_t i) const { return u[i] - v[i]; }
  size_t Size() const { return v.Size(); }
};
template <typename S, typename T>
VectorSubtraction(const S&, const T&) -> VectorSubtraction<S, T>;

template <typename S, typename T>
auto operator-(const VectorExpr<S>& u, const VectorExpr<T>& v)
{
  return VectorSubtraction(u.asDerived(), v.asDerived());
}

template <typename T>
auto operator-(const mfem::Vector& u, const VectorExpr<T>& v)
{
  return VectorSubtraction(u, v.asDerived());
}

template <typename T>
auto operator-(const VectorExpr<T>& u, const mfem::Vector& v)
{
  return VectorSubtraction(u.asDerived(), v);
}

auto operator-(const mfem::Vector& u, const mfem::Vector& v) { return VectorSubtraction(u, v); }

template <typename vec>
struct Matvec : VectorExpr<Matvec<vec> > {
  const mfem::Operator& A;
  const vec&            v;
  mfem::Vector          result;
  Matvec(const mfem::Operator& A, const vec& v) : A(A), v(v)
  {
    result.SetSize(A.Height());
    if constexpr (std::is_same<vec, mfem::Vector>::value) {
      A.Mult(v, result);
    } else {
      mfem::Vector tmp = evaluate(v);
      A.Mult(tmp, result);
    }
  }
  double operator[](size_t i) const { return result[i]; }
  size_t Size() const { return result.Size(); }
};
template <typename T>
Matvec(const mfem::Operator& A, const T&) -> Matvec<T>;

template <typename T>
auto operator*(const mfem::Operator& A, const VectorExpr<T>& v)
{
  return Matvec(A, v.asDerived());
}

auto operator*(const mfem::Operator& A, const mfem::Vector& v) { return Matvec(A, v); }

#endif

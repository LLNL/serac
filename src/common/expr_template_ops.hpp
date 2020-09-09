// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file expr_template_ops.hpp
 *
 * @brief The operator overloads for mfem::Vector
 */

#ifndef EXPR_TEMPLATE_OPS
#define EXPR_TEMPLATE_OPS

#include "common/expr_template_internal.hpp"

template <typename T>
auto operator-(serac::VectorExpr<T>&& u)
{
  return serac::internal::UnaryNegation<T, true>(std::move(u.asDerived()));
}

inline auto operator-(const mfem::Vector& u) { return serac::internal::UnaryNegation<mfem::Vector, false>(u); }

inline auto operator-(mfem::Vector&& u) { return serac::internal::UnaryNegation<mfem::Vector, true>(std::move(u)); }

template <typename T>
auto operator*(serac::VectorExpr<T>&& u, const double a)
{
  using serac::internal::ScalarMultOp;
  return serac::internal::UnaryVectorExpr<T, true, ScalarMultOp>(std::move(u.asDerived()), ScalarMultOp{a});
}

template <typename T>
auto operator*(const double a, serac::VectorExpr<T>&& u)
{
  return operator*(std::move(u), a);
}

inline auto operator*(const double a, const mfem::Vector& u)
{
  using serac::internal::ScalarMultOp;
  return serac::internal::UnaryVectorExpr<mfem::Vector, false, ScalarMultOp>(u, ScalarMultOp{a});
}

inline auto operator*(const mfem::Vector& u, const double a) { return operator*(a, u); }

inline auto operator*(const double a, mfem::Vector&& u)
{
  using serac::internal::ScalarMultOp;
  return serac::internal::UnaryVectorExpr<mfem::Vector, true, ScalarMultOp>(std::move(u), ScalarMultOp{a});
}

inline auto operator*(mfem::Vector&& u, const double a) { return operator*(a, std::move(u)); }

template <typename S, typename T>
auto operator+(serac::VectorExpr<S>&& u, serac::VectorExpr<T>&& v)
{
  return serac::internal::VectorAddition<S, T, true, true>(std::move(u.asDerived()), std::move(v.asDerived()));
}

template <typename T>
auto operator+(const mfem::Vector& u, serac::VectorExpr<T>&& v)
{
  return serac::internal::VectorAddition<mfem::Vector, T, false, true>(u, std::move(v.asDerived()));
}

template <typename T>
auto operator+(serac::VectorExpr<T>&& u, const mfem::Vector& v)
{
  return operator+(v, std::move(u));
}

inline auto operator+(const mfem::Vector& u, const mfem::Vector& v)
{
  return serac::internal::VectorAddition<mfem::Vector, mfem::Vector, false, false>(u, v);
}

template <typename T>
auto operator+(mfem::Vector&& u, serac::VectorExpr<T>&& v)
{
  return serac::internal::VectorAddition<mfem::Vector, T, true, true>(std::move(u), std::move(v.asDerived()));
}

template <typename T>
auto operator+(serac::VectorExpr<T>&& u, mfem::Vector&& v)
{
  return operator+(std::move(v), std::move(u));
}

inline auto operator+(mfem::Vector&& u, mfem::Vector&& v)
{
  return serac::internal::VectorAddition<mfem::Vector, mfem::Vector, true, true>(std::move(u), std::move(v));
}

inline auto operator+(const mfem::Vector& u, mfem::Vector&& v)
{
  return serac::internal::VectorAddition<mfem::Vector, mfem::Vector, false, true>(u, std::move(v));
}

inline auto operator+(mfem::Vector&& u, const mfem::Vector& v) { return operator+(v, std::move(u)); }

template <typename S, typename T>
auto operator-(serac::VectorExpr<S>&& u, serac::VectorExpr<T>&& v)
{
  return serac::internal::VectorSubtraction<S, T, true, true>(std::move(u.asDerived()), std::move(v.asDerived()));
}

template <typename T>
auto operator-(const mfem::Vector& u, serac::VectorExpr<T>&& v)
{
  return serac::internal::VectorSubtraction<mfem::Vector, T, false, true>(u, std::move(v.asDerived()));
}

template <typename T>
auto operator-(serac::VectorExpr<T>&& u, const mfem::Vector& v)
{
  return serac::internal::VectorSubtraction<T, mfem::Vector, true, false>(std::move(u.asDerived()), v);
}

inline auto operator-(const mfem::Vector& u, const mfem::Vector& v)
{
  return serac::internal::VectorSubtraction<mfem::Vector, mfem::Vector, false, false>(u, v);
}

template <typename T>
auto operator-(mfem::Vector&& u, serac::VectorExpr<T>&& v)
{
  return serac::internal::VectorSubtraction<mfem::Vector, T, true, true>(std::move(u), std::move(v.asDerived()));
}

template <typename T>
auto operator-(serac::VectorExpr<T>&& u, mfem::Vector&& v)
{
  return serac::internal::VectorSubtraction<T, mfem::Vector, true, true>(std::move(u.asDerived()), std::move(v));
}

inline auto operator-(mfem::Vector&& u, mfem::Vector&& v)
{
  return serac::internal::VectorSubtraction<mfem::Vector, mfem::Vector, true, true>(std::move(u), std::move(v));
}

inline auto operator-(const mfem::Vector& u, mfem::Vector&& v)
{
  return serac::internal::VectorSubtraction<mfem::Vector, mfem::Vector, false, true>(u, std::move(v));
}

inline auto operator-(mfem::Vector&& u, const mfem::Vector& v) { return operator-(v, std::move(u)); }

template <typename T>
auto operator*(const mfem::Operator& A, serac::VectorExpr<T>&& v)
{
  return serac::internal::OperatorExpr<T>(A, std::move(v.asDerived()));
}

inline auto operator*(const mfem::Operator& A, const mfem::Vector& v)
{
  return serac::internal::OperatorExpr<mfem::Vector>(A, v);
}

#endif

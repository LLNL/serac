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
auto operator-(VectorExpr<T>&& u)
{
  return serac::internal::UnaryNegation<T, true>(std::move(u.asDerived()));
}

inline auto operator-(const mfem::Vector& u) { return serac::internal::UnaryNegation<mfem::Vector, false>(u); }

template <typename T>
auto operator*(VectorExpr<T>&& u, const double a)
{
  using serac::internal::ScalarMultOp;
  return serac::internal::UnaryVectorExpr<T, true, ScalarMultOp>(std::move(u.asDerived()), ScalarMultOp{a});
}

template <typename T>
auto operator*(const double a, VectorExpr<T>&& u)
{
  return operator*(std::move(u), a);
}

inline auto operator*(const double a, const mfem::Vector& u)
{
  using serac::internal::ScalarMultOp;
  return serac::internal::UnaryVectorExpr<mfem::Vector, false, ScalarMultOp>(u, ScalarMultOp{a});
}

inline auto operator*(const mfem::Vector& u, const double a) { return operator*(a, u); }

template <typename S, typename T>
auto operator+(VectorExpr<S>&& u, VectorExpr<T>&& v)
{
  return serac::internal::VectorAddition<S, T, true, true>(std::move(u.asDerived()), std::move(v.asDerived()));
}

template <typename T>
auto operator+(const mfem::Vector& u, VectorExpr<T>&& v)
{
  return serac::internal::VectorAddition<mfem::Vector, T, false, true>(u, std::move(v.asDerived()));
}

template <typename T>
auto operator+(VectorExpr<T>&& u, const mfem::Vector& v)
{
  return operator+(v, std::move(u));
}

inline auto operator+(const mfem::Vector& u, const mfem::Vector& v)
{
  return serac::internal::VectorAddition<mfem::Vector, mfem::Vector, false, false>(u, v);
}

template <typename S, typename T>
auto operator-(VectorExpr<S>&& u, VectorExpr<T>&& v)
{
  return serac::internal::VectorSubtraction<S, T, true, true>(std::move(u.asDerived()), std::move(v.asDerived()));
}

template <typename T>
auto operator-(const mfem::Vector& u, VectorExpr<T>&& v)
{
  return serac::internal::VectorSubtraction<mfem::Vector, T, false, true>(u, std::move(v.asDerived()));
}

template <typename T>
auto operator-(VectorExpr<T>&& u, const mfem::Vector& v)
{
  return operator-(v, std::move(u));
}

inline auto operator-(const mfem::Vector& u, const mfem::Vector& v)
{
  return serac::internal::VectorSubtraction<mfem::Vector, mfem::Vector, false, false>(u, v);
}

template <typename T>
auto operator*(const mfem::Operator& A, VectorExpr<T>&& v)
{
  return serac::internal::Matvec<T, true>(A, std::move(v.asDerived()));
}

inline auto operator*(const mfem::Operator& A, const mfem::Vector& v)
{
  return serac::internal::Matvec<mfem::Vector, false>(A, v);
}

#endif
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

#pragma once

#include "serac/numerics/expr_template_impl.hpp"

template <typename T>
auto operator-(serac::VectorExpr<T>&& u)
{
  return serac::detail::UnaryNegation<T>(std::move(u.asDerived()));
}

template <typename MFEMVec, typename = serac::detail::enable_if_mfem_vec<MFEMVec>>
auto operator-(MFEMVec&& u)
{
  return serac::detail::UnaryNegation<MFEMVec>(std::forward<MFEMVec>(u));
}

template <typename T>
auto operator*(serac::VectorExpr<T>&& u, const double a)
{
  using serac::detail::ScalarMultOp;
  return serac::detail::UnaryVectorExpr<T, ScalarMultOp>(std::move(u.asDerived()), ScalarMultOp{a});
}

template <typename T>
auto operator*(const double a, serac::VectorExpr<T>&& u)
{
  return operator*(std::move(u), a);
}

template <typename MFEMVec, typename = serac::detail::enable_if_mfem_vec<MFEMVec>>
auto operator*(const double a, MFEMVec&& u)
{
  using serac::detail::ScalarMultOp;
  return serac::detail::UnaryVectorExpr<MFEMVec, ScalarMultOp>(std::forward<MFEMVec>(u), ScalarMultOp{a});
}

template <typename MFEMVec, typename = serac::detail::enable_if_mfem_vec<MFEMVec>>
auto operator*(MFEMVec&& u, const double a)
{
  return operator*(a, std::forward<MFEMVec>(u));
}

template <typename T>
auto operator/(serac::VectorExpr<T>&& u, const double a)
{
  using serac::detail::ScalarDivOp;
  return serac::detail::UnaryVectorExpr<T, ScalarDivOp<true>>(std::move(u.asDerived()), ScalarDivOp{a});
}

template <typename MFEMVec, typename = serac::detail::enable_if_mfem_vec<MFEMVec>>
auto operator/(MFEMVec&& u, const double a)
{
  using serac::detail::ScalarDivOp;
  return serac::detail::UnaryVectorExpr<MFEMVec, ScalarDivOp<true>>(std::forward<MFEMVec>(u), ScalarDivOp{a});
}

template <typename T>
auto operator/(const double a, serac::VectorExpr<T>&& u)
{
  using serac::detail::ScalarDivOp;
  return serac::detail::UnaryVectorExpr<T, ScalarDivOp<false>>(std::move(u.asDerived()), ScalarDivOp<false>{a});
}

template <typename MFEMVec, typename = serac::detail::enable_if_mfem_vec<MFEMVec>>
auto operator/(const double a, MFEMVec&& u)
{
  using serac::detail::ScalarDivOp;
  return serac::detail::UnaryVectorExpr<MFEMVec, ScalarDivOp<false>>(std::forward<MFEMVec>(u), ScalarDivOp<false>{a});
}

template <typename S, typename T>
auto operator+(serac::VectorExpr<S>&& u, serac::VectorExpr<T>&& v)
{
  return serac::detail::VectorAddition<S, T>(std::move(u.asDerived()), std::move(v.asDerived()));
}

template <typename T, typename MFEMVec, typename = serac::detail::enable_if_mfem_vec<MFEMVec>>
auto operator+(MFEMVec&& u, serac::VectorExpr<T>&& v)
{
  return serac::detail::VectorAddition<MFEMVec, T>(std::forward<MFEMVec>(u), std::move(v.asDerived()));
}

template <typename T, typename MFEMVec, typename = serac::detail::enable_if_mfem_vec<MFEMVec>>
auto operator+(serac::VectorExpr<T>&& u, MFEMVec&& v)
{
  return operator+(std::forward<MFEMVec>(v), std::move(u));
}

template <typename MFEMVecL, typename MFEMVecR, typename = serac::detail::enable_if_mfem_vec<MFEMVecL>,
          typename = serac::detail::enable_if_mfem_vec<MFEMVecR>>
auto operator+(MFEMVecL&& u, MFEMVecR&& v)
{
  return serac::detail::VectorAddition<MFEMVecL, MFEMVecR>(std::forward<MFEMVecL>(u), std::forward<MFEMVecR>(v));
}

template <typename S, typename T>
auto operator-(serac::VectorExpr<S>&& u, serac::VectorExpr<T>&& v)
{
  return serac::detail::VectorSubtraction<S, T>(std::move(u.asDerived()), std::move(v.asDerived()));
}

template <typename T, typename MFEMVec, typename = serac::detail::enable_if_mfem_vec<MFEMVec>>
auto operator-(MFEMVec&& u, serac::VectorExpr<T>&& v)
{
  return serac::detail::VectorSubtraction<MFEMVec, T>(std::forward<MFEMVec>(u), std::move(v.asDerived()));
}

template <typename T, typename MFEMVec, typename = serac::detail::enable_if_mfem_vec<MFEMVec>>
auto operator-(serac::VectorExpr<T>&& u, MFEMVec&& v)
{
  return serac::detail::VectorSubtraction<T, MFEMVec>(std::move(u.asDerived()), std::forward<MFEMVec>(v));
}

template <typename MFEMVecL, typename MFEMVecR, typename = serac::detail::enable_if_mfem_vec<MFEMVecL>,
          typename = serac::detail::enable_if_mfem_vec<MFEMVecR>>
auto operator-(MFEMVecL&& u, MFEMVecR&& v)
{
  return serac::detail::VectorSubtraction<MFEMVecL, MFEMVecR>(std::forward<MFEMVecL>(u), std::forward<MFEMVecR>(v));
}

template <typename T>
auto operator*(const mfem::Operator& A, serac::VectorExpr<T>&& v)
{
  return serac::detail::OperatorExpr<T>(A, std::move(v.asDerived()));
}

inline auto operator*(const mfem::Operator& A, const mfem::Vector& v)
{
  return serac::detail::OperatorExpr<mfem::Vector>(A, v);
}

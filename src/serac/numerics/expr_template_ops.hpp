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
  return detail::UnaryNegation<T>(std::move(u.asDerived()));
}

template <typename MFEMVec, typename = detail::enable_if_mfem_vec<MFEMVec>>
auto operator-(MFEMVec&& u)
{
  return detail::UnaryNegation<MFEMVec>(std::forward<MFEMVec>(u));
}

template <typename T>
auto operator*(serac::VectorExpr<T>&& u, const double a)
{
  using detail::ScalarMultOp;
  return detail::UnaryVectorExpr<T, ScalarMultOp>(std::move(u.asDerived()), ScalarMultOp{a});
}

template <typename T>
auto operator*(const double a, serac::VectorExpr<T>&& u)
{
  return operator*(std::move(u), a);
}

template <typename MFEMVec, typename = detail::enable_if_mfem_vec<MFEMVec>>
auto operator*(const double a, MFEMVec&& u)
{
  using detail::ScalarMultOp;
  return detail::UnaryVectorExpr<MFEMVec, ScalarMultOp>(std::forward<MFEMVec>(u), ScalarMultOp{a});
}

template <typename MFEMVec, typename = detail::enable_if_mfem_vec<MFEMVec>>
auto operator*(MFEMVec&& u, const double a)
{
  return operator*(a, std::forward<MFEMVec>(u));
}

template <typename T>
auto operator/(serac::VectorExpr<T>&& u, const double a)
{
  using detail::ScalarDivOp;
  return detail::UnaryVectorExpr<T, ScalarDivOp<true>>(std::move(u.asDerived()), ScalarDivOp{a});
}

template <typename MFEMVec, typename = detail::enable_if_mfem_vec<MFEMVec>>
auto operator/(MFEMVec&& u, const double a)
{
  using detail::ScalarDivOp;
  return detail::UnaryVectorExpr<MFEMVec, ScalarDivOp<true>>(std::forward<MFEMVec>(u), ScalarDivOp{a});
}

template <typename T>
auto operator/(const double a, serac::VectorExpr<T>&& u)
{
  using detail::ScalarDivOp;
  return detail::UnaryVectorExpr<T, ScalarDivOp<false>>(std::move(u.asDerived()), ScalarDivOp<false>{a});
}

template <typename MFEMVec, typename = detail::enable_if_mfem_vec<MFEMVec>>
auto operator/(const double a, MFEMVec&& u)
{
  using detail::ScalarDivOp;
  return detail::UnaryVectorExpr<MFEMVec, ScalarDivOp<false>>(std::forward<MFEMVec>(u), ScalarDivOp<false>{a});
}

template <typename S, typename T>
auto operator+(serac::VectorExpr<S>&& u, serac::VectorExpr<T>&& v)
{
  return detail::VectorAddition<S, T>(std::move(u.asDerived()), std::move(v.asDerived()));
}

template <typename T, typename MFEMVec, typename = detail::enable_if_mfem_vec<MFEMVec>>
auto operator+(MFEMVec&& u, serac::VectorExpr<T>&& v)
{
  return detail::VectorAddition<MFEMVec, T>(std::forward<MFEMVec>(u), std::move(v.asDerived()));
}

template <typename T, typename MFEMVec, typename = detail::enable_if_mfem_vec<MFEMVec>>
auto operator+(serac::VectorExpr<T>&& u, MFEMVec&& v)
{
  return operator+(std::forward<MFEMVec>(v), std::move(u));
}

template <typename MFEMVecL, typename MFEMVecR, typename = detail::enable_if_mfem_vec<MFEMVecL>,
          typename = detail::enable_if_mfem_vec<MFEMVecR>>
auto operator+(MFEMVecL&& u, MFEMVecR&& v)
{
  return detail::VectorAddition<MFEMVecL, MFEMVecR>(std::forward<MFEMVecL>(u), std::forward<MFEMVecR>(v));
}

template <typename S, typename T>
auto operator-(serac::VectorExpr<S>&& u, serac::VectorExpr<T>&& v)
{
  return detail::VectorSubtraction<S, T>(std::move(u.asDerived()), std::move(v.asDerived()));
}

template <typename T, typename MFEMVec, typename = detail::enable_if_mfem_vec<MFEMVec>>
auto operator-(MFEMVec&& u, serac::VectorExpr<T>&& v)
{
  return detail::VectorSubtraction<MFEMVec, T>(std::forward<MFEMVec>(u), std::move(v.asDerived()));
}

template <typename T, typename MFEMVec, typename = detail::enable_if_mfem_vec<MFEMVec>>
auto operator-(serac::VectorExpr<T>&& u, MFEMVec&& v)
{
  return detail::VectorSubtraction<T, MFEMVec>(std::move(u.asDerived()), std::forward<MFEMVec>(v));
}

template <typename MFEMVecL, typename MFEMVecR, typename = detail::enable_if_mfem_vec<MFEMVecL>,
          typename = detail::enable_if_mfem_vec<MFEMVecR>>
auto operator-(MFEMVecL&& u, MFEMVecR&& v)
{
  return detail::VectorSubtraction<MFEMVecL, MFEMVecR>(std::forward<MFEMVecL>(u), std::forward<MFEMVecR>(v));
}

template <typename T>
auto operator*(const mfem::Operator& A, serac::VectorExpr<T>&& v)
{
  return detail::OperatorExpr<T>(A, std::move(v.asDerived()));
}

inline auto operator*(const mfem::Operator& A, const mfem::Vector& v)
{
  return detail::OperatorExpr<mfem::Vector>(A, v);
}

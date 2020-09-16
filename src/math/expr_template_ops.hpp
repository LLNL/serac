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

#include "math/expr_template_internal.hpp"

/**
 * @brief A utility for conditionally enabling templates if a given type
 * is an mfem::Vector
 */
template <typename MFEMVec>
using enable_if_mfem_vec = std::enable_if_t<std::is_same_v<std::decay_t<MFEMVec>, mfem::Vector>>;

template <typename T>
auto operator-(serac::VectorExpr<T>&& u)
{
  return serac::internal::UnaryNegation<T>(std::move(u.asDerived()));
}

template <typename MFEMVec, typename = enable_if_mfem_vec<MFEMVec>>
auto operator-(MFEMVec&& u)
{
  return serac::internal::UnaryNegation<MFEMVec>(std::forward<MFEMVec>(u));
}

template <typename T>
auto operator*(serac::VectorExpr<T>&& u, const double a)
{
  using serac::internal::ScalarMultOp;
  return serac::internal::UnaryVectorExpr<T, ScalarMultOp>(std::move(u.asDerived()), ScalarMultOp{a});
}

template <typename T>
auto operator*(const double a, serac::VectorExpr<T>&& u)
{
  return operator*(std::move(u), a);
}

template <typename MFEMVec, typename = enable_if_mfem_vec<MFEMVec>>
auto operator*(const double a, MFEMVec&& u)
{
  using serac::internal::ScalarMultOp;
  return serac::internal::UnaryVectorExpr<MFEMVec, ScalarMultOp>(std::forward<MFEMVec>(u), ScalarMultOp{a});
}

template <typename MFEMVec, typename = enable_if_mfem_vec<MFEMVec>>
auto operator*(MFEMVec&& u, const double a)
{
  return operator*(a, std::forward<MFEMVec>(u));
}

template <typename T>
auto operator/(serac::VectorExpr<T>&& u, const double a)
{
  using serac::internal::ScalarDivOp;
  return serac::internal::UnaryVectorExpr<T, ScalarDivOp<true>>(std::move(u.asDerived()), ScalarDivOp{a});
}

template <typename MFEMVec, typename = enable_if_mfem_vec<MFEMVec>>
auto operator/(MFEMVec&& u, const double a)
{
  using serac::internal::ScalarDivOp;
  return serac::internal::UnaryVectorExpr<MFEMVec, ScalarDivOp<true>>(std::forward<MFEMVec>(u), ScalarDivOp{a});
}

template <typename T>
auto operator/(const double a, serac::VectorExpr<T>&& u)
{
  using serac::internal::ScalarDivOp;
  return serac::internal::UnaryVectorExpr<T, ScalarDivOp<false>>(std::move(u.asDerived()), ScalarDivOp<false>{a});
}

template <typename MFEMVec, typename = enable_if_mfem_vec<MFEMVec>>
auto operator/(const double a, MFEMVec&& u)
{
  using serac::internal::ScalarDivOp;
  return serac::internal::UnaryVectorExpr<MFEMVec, ScalarDivOp<false>>(std::forward<MFEMVec>(u), ScalarDivOp<false>{a});
}

template <typename S, typename T>
auto operator+(serac::VectorExpr<S>&& u, serac::VectorExpr<T>&& v)
{
  return serac::internal::VectorAddition<S, T>(std::move(u.asDerived()), std::move(v.asDerived()));
}

template <typename T, typename MFEMVec, typename = enable_if_mfem_vec<MFEMVec>>
auto operator+(MFEMVec&& u, serac::VectorExpr<T>&& v)
{
  return serac::internal::VectorAddition<MFEMVec, T>(std::forward<MFEMVec>(u), std::move(v.asDerived()));
}

template <typename T, typename MFEMVec, typename = enable_if_mfem_vec<MFEMVec>>
auto operator+(serac::VectorExpr<T>&& u, MFEMVec&& v)
{
  return operator+(std::forward<MFEMVec>(v), std::move(u));
}

template <typename MFEMVecL, typename MFEMVecR, typename = enable_if_mfem_vec<MFEMVecL>,
          typename = enable_if_mfem_vec<MFEMVecR>>
auto operator+(MFEMVecL&& u, MFEMVecR&& v)
{
  return serac::internal::VectorAddition<MFEMVecL, MFEMVecR>(std::forward<MFEMVecL>(u), std::forward<MFEMVecR>(v));
}

template <typename S, typename T>
auto operator-(serac::VectorExpr<S>&& u, serac::VectorExpr<T>&& v)
{
  return serac::internal::VectorSubtraction<S, T>(std::move(u.asDerived()), std::move(v.asDerived()));
}

template <typename T, typename MFEMVec, typename = enable_if_mfem_vec<MFEMVec>>
auto operator-(MFEMVec&& u, serac::VectorExpr<T>&& v)
{
  return serac::internal::VectorSubtraction<MFEMVec, T>(std::forward<MFEMVec>(u), std::move(v.asDerived()));
}

template <typename T, typename MFEMVec, typename = enable_if_mfem_vec<MFEMVec>>
auto operator-(serac::VectorExpr<T>&& u, MFEMVec&& v)
{
  return serac::internal::VectorSubtraction<T, MFEMVec>(std::move(u.asDerived()), std::forward<MFEMVec>(v));
}

template <typename MFEMVecL, typename MFEMVecR, typename = enable_if_mfem_vec<MFEMVecL>,
          typename = enable_if_mfem_vec<MFEMVecR>>
auto operator-(MFEMVecL&& u, MFEMVecR&& v)
{
  return serac::internal::VectorSubtraction<MFEMVecL, MFEMVecR>(std::forward<MFEMVecL>(u), std::forward<MFEMVecR>(v));
}

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

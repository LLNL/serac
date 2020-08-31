// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include "common/expr_template_ops.hpp"

TEST(expr_templates, basic_add)
{
  constexpr int size = 10;
  mfem::Vector  lhs(size);
  mfem::Vector  rhs(size);
  for (int i = 0; i < size; i++) {
    lhs[i] = i * 4 + 1;
    rhs[i] = i * i * 3 + 2;
  }

  mfem::Vector mfem_result(size);
  add(lhs, rhs, mfem_result);

  mfem::Vector expr_result = lhs + rhs;

  for (int i = 0; i < size; i++) {
    EXPECT_FLOAT_EQ(mfem_result[i], expr_result[i]);
  }
}

TEST(expr_templates, basic_add_lambda)
{
  constexpr int size = 10;
  mfem::Vector  lhs(size);
  mfem::Vector  rhs(size);
  for (int i = 0; i < size; i++) {
    lhs[i] = i * 4 + 1;
    rhs[i] = i * i * 3 + 2;
  }

  mfem::Vector mfem_result(size);
  add(lhs, rhs, mfem_result);

  auto lambda_add = [](const auto& lhs, const auto& rhs) { return lhs + rhs; };

  mfem::Vector expr_result = lambda_add(lhs, rhs);

  EXPECT_EQ(mfem_result.Size(), expr_result.Size());
  for (int i = 0; i < size; i++) {
    EXPECT_FLOAT_EQ(mfem_result[i], expr_result[i]);
  }
}

TEST(expr_templates, move_from_temp_lambda)
{
  constexpr int size = 10;
  mfem::Vector  lhs(size);
  mfem::Vector  rhs(size);
  for (int i = 0; i < size; i++) {
    lhs[i] = i * 4 + 1;
    rhs[i] = i * i * 3 + 2;
  }

  mfem::Vector mfem_result(size);
  add(lhs, 3.5, rhs, mfem_result);

  auto lambda_add = [](const auto& lhs, const auto& rhs) {
    auto r35 = rhs * 3.5;
    return lhs + std::move(r35);
  };

  mfem::Vector expr_result = lambda_add(lhs, rhs);

  EXPECT_EQ(mfem_result.Size(), expr_result.Size());
  for (int i = 0; i < size; i++) {
    EXPECT_FLOAT_EQ(mfem_result[i], expr_result[i]);
  }
}

TEST(expr_templates, basic_matvec)
{
  constexpr int     rows = 10;
  constexpr int     cols = 12;
  mfem::Vector      vec_in(cols);
  mfem::DenseMatrix matrix(rows, cols);
  for (int i = 0; i < cols; i++) {
    vec_in[i] = i * 4 + 1;
    for (int j = 0; j < rows; j++) {
      matrix(j, i) = 2 * (i == j) - (i == (j + 1)) - (i == (j - 1));
    }
  }

  mfem::Vector mfem_result(rows);
  matrix.Mult(vec_in, mfem_result);

  mfem::Vector expr_result = matrix * vec_in;

  EXPECT_EQ(mfem_result.Size(), expr_result.Size());
  for (int i = 0; i < cols; i++) {
    EXPECT_FLOAT_EQ(mfem_result[i], expr_result[i]);
  }
}

TEST(expr_templates, complex_expr)
{
  constexpr int rows = 10;
  mfem::Vector  lhs(rows);
  mfem::Vector  rhs(rows);
  for (int i = 0; i < rows; i++) {
    lhs[i] = i * 4 + 1;
    rhs[i] = i * i * 3 + 2;
  }

  constexpr int     cols = 12;
  mfem::Vector      vec_in(cols);
  mfem::DenseMatrix matrix(rows, cols);
  for (int i = 0; i < cols; i++) {
    vec_in[i] = i * 4 + 1;
    for (int j = 0; j < rows; j++) {
      matrix(j, i) = 2 * (i == j) - (i == (j + 1)) - (i == (j - 1));
    }
  }

  mfem::Vector matvec(rows);
  matrix.Mult(vec_in, matvec);

  mfem::Vector vec_negate_scale(rows);
  add(-1.0, lhs, 3.0, rhs, vec_negate_scale);

  mfem::Vector mfem_result(rows);
  add(vec_negate_scale, -0.3, matvec, mfem_result);

  mfem::Vector expr_result = -lhs + rhs * 3.0 - 0.3 * (matrix * vec_in);

  EXPECT_EQ(mfem_result.Size(), expr_result.Size());
  for (int i = 0; i < cols; i++) {
    EXPECT_FLOAT_EQ(mfem_result[i], expr_result[i]);
  }
}

TEST(expr_templates, complex_expr_lambda)
{
  constexpr int rows = 10;
  mfem::Vector  lhs(rows);
  mfem::Vector  rhs(rows);
  for (int i = 0; i < rows; i++) {
    lhs[i] = i * 4 + 1;
    rhs[i] = i * i * 3 + 2;
  }

  constexpr int     cols = 12;
  mfem::Vector      vec_in(cols);
  mfem::DenseMatrix matrix(rows, cols);
  for (int i = 0; i < cols; i++) {
    vec_in[i] = i * 4 + 1;
    for (int j = 0; j < rows; j++) {
      matrix(j, i) = 2 * (i == j) - (i == (j + 1)) - (i == (j - 1));
    }
  }

  mfem::Vector matvec(rows);
  matrix.Mult(vec_in, matvec);

  mfem::Vector vec_negate_scale(rows);
  add(-1.0, lhs, 3.0, rhs, vec_negate_scale);

  mfem::Vector mfem_result(rows);
  add(vec_negate_scale, -0.3, matvec, mfem_result);

  auto lambda_expr = [](const auto& lhs, const auto& rhs, const auto& matrix, const auto& vec_in) {
    return -lhs + rhs * 3.0 - 0.3 * (matrix * vec_in);
  };

  mfem::Vector expr_result = lambda_expr(lhs, rhs, matrix, vec_in);

  EXPECT_EQ(mfem_result.Size(), expr_result.Size());
  for (int i = 0; i < cols; i++) {
    EXPECT_FLOAT_EQ(mfem_result[i], expr_result[i]);
  }
}
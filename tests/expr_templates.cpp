// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include <chrono>
#include <iostream>

#include "common/expr_template_ops.hpp"

TEST(expr_templates, basic_add)
{
  MPI_Barrier(MPI_COMM_WORLD);
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
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(expr_templates, basic_add_parallel)
{
  MPI_Barrier(MPI_COMM_WORLD);
  constexpr int size = 10;
  mfem::Vector  lhs(size);
  mfem::Vector  rhs(size);
  for (int i = 0; i < size; i++) {
    lhs[i] = i * 4 + 1;
    rhs[i] = i * i * 3 + 2;
  }

  mfem::Vector mfem_result(size);
  add(lhs, rhs, mfem_result);

  mfem::Vector expr_result(size);
  evaluate(lhs + rhs, expr_result, MPI_COMM_WORLD);

  for (int i = 0; i < size; i++) {
    EXPECT_FLOAT_EQ(mfem_result[i], expr_result[i]);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(expr_templates, basic_div)
{
  MPI_Barrier(MPI_COMM_WORLD);
  constexpr int    size   = 10;
  constexpr double scalar = 0.3;
  mfem::Vector     a(size);
  mfem::Vector     mfem_result(size);
  for (int i = 0; i < size; i++) {
    a[i]           = i * 4 + 1;
    mfem_result[i] = a[i];
  }

  // Dividing a vector by a scalar
  mfem_result /= scalar;

  mfem::Vector expr_result = a / scalar;

  for (int i = 0; i < size; i++) {
    EXPECT_FLOAT_EQ(mfem_result[i], expr_result[i]);
  }

  // Dividing a scalar by a vector
  mfem_result = a;
  for (int i = 0; i < size; i++) {
    a[i] = scalar / a[i];
  }

  expr_result = scalar / a;

  for (int i = 0; i < size; i++) {
    EXPECT_FLOAT_EQ(mfem_result[i], expr_result[i]);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(expr_templates, basic_add_lambda)
{
  MPI_Barrier(MPI_COMM_WORLD);
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
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(expr_templates, subtraction_not_commutative)
{
  MPI_Barrier(MPI_COMM_WORLD);
  constexpr int size = 10;
  mfem::Vector  a(size);
  mfem::Vector  b(size);
  mfem::Vector  c(size);

  for (int i = 0; i < size; i++) {
    a[i] = i * 4 + 1;
    b[i] = i * i * 3 + 2;
    c[i] = i * i * i * 7 + 23;
  }

  // Tests that switching the order of operations
  // does not change the result
  mfem::Vector result1 = a + c - b;  // Parsed as (a + c) - b
  mfem::Vector result2 = c - b + a;  // Parsed as (c - b) + a
  for (int i = 0; i < size; i++) {
    EXPECT_FLOAT_EQ(result1[i], result2[i]);
  }

  mfem::Vector result3 = a - c;
  mfem::Vector result4 = c - a;
  for (int i = 0; i < size; i++) {
    EXPECT_FALSE(result3[i] == result4[i]);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(expr_templates, subtraction_not_commutative_rvalue)
{
  MPI_Barrier(MPI_COMM_WORLD);
  constexpr int size           = 3;
  double        a_values[size] = {-12.2692, 6.23918, -12.2692};
  double        b_values[size] = {0.0850848, -0.17017, 0.0850848};

  auto fext = [](const double t) {
    mfem::Vector force(3);
    force[0] = -10 * t;
    force[1] = 0;
    force[2] = 10 * t * t;
    return force;
  };

  mfem::Vector a(a_values, size);
  mfem::Vector b(b_values, size);
  mfem::Vector c        = fext(0.0);
  mfem::Vector resulta1 = c - a - b;
  mfem::Vector resulta2 = fext(0.0) - a - b;
  for (int i = 0; i < size; i++) {
    EXPECT_FLOAT_EQ(resulta1[i], resulta2[i]);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(expr_templates, addition_commutative)
{
  MPI_Barrier(MPI_COMM_WORLD);
  constexpr int size = 10;
  mfem::Vector  a(size);
  mfem::Vector  b(size);

  for (int i = 0; i < size; i++) {
    a[i] = i * 4 + 1;
    b[i] = i * i * 3 + 2;
  }

  mfem::Vector result1 = a + b;
  mfem::Vector result2 = b + a;
  for (int i = 0; i < size; i++) {
    EXPECT_FLOAT_EQ(result1[i], result2[i]);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(expr_templates, scalar_mult_commutative)
{
  MPI_Barrier(MPI_COMM_WORLD);
  constexpr int size = 10;
  mfem::Vector  a(size);
  double        scalar = 0.3;

  for (int i = 0; i < size; i++) {
    a[i] = i * 4 + 1;
  }

  mfem::Vector result1 = scalar * a;
  mfem::Vector result2 = a * scalar;
  for (int i = 0; i < size; i++) {
    EXPECT_FLOAT_EQ(result1[i], result2[i]);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(expr_templates, move_from_temp_lambda)
{
  MPI_Barrier(MPI_COMM_WORLD);
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
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(expr_templates, move_from_temp_vec_lambda)
{
  MPI_Barrier(MPI_COMM_WORLD);
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
    mfem::Vector r35 = rhs * 3.5;
    return lhs + std::move(r35);
  };

  mfem::Vector expr_result = lambda_add(lhs, rhs);

  EXPECT_EQ(mfem_result.Size(), expr_result.Size());
  for (int i = 0; i < size; i++) {
    EXPECT_FLOAT_EQ(mfem_result[i], expr_result[i]);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(expr_templates, small_matvec)
{
  MPI_Barrier(MPI_COMM_WORLD);
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
  for (int i = 0; i < rows; i++) {
    EXPECT_FLOAT_EQ(mfem_result[i], expr_result[i]);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(expr_templates, small_mixed_expr)
{
  MPI_Barrier(MPI_COMM_WORLD);
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

  constexpr int trials     = 100;
  double        mfem_total = 0.0;
  double        expr_total = 0.0;

  mfem::Vector mfem_result(rows);
  mfem::Vector expr_result(rows);

  // -lhs + rhs * 3.0 - 0.3 * (matrix * vec_in)
  for (int i = 0; i < trials; i++) {
    auto mfem_start = std::chrono::steady_clock::now();
    for (int j = 0; j < trials; j++) {
      mfem::Vector matvec(rows);
      matrix.Mult(vec_in, matvec);

      mfem::Vector vec_negate_scale(rows);
      add(-1.0, lhs, 3.0, rhs, vec_negate_scale);

      add(vec_negate_scale, -0.3, matvec, mfem_result);
    }
    mfem_total += (std::chrono::steady_clock::now() - mfem_start).count();
    auto expr_start = std::chrono::steady_clock::now();
    for (int j = 0; j < trials; j++) {
      expr_result = -lhs + rhs * 3.0 - 0.3 * (matrix * vec_in);
    }
    expr_total += (std::chrono::steady_clock::now() - expr_start).count();
  }
  const auto mfem_avg = mfem_total / trials;
  const auto expr_avg = expr_total / trials;
  std::cout << "Expression templates took " << (expr_avg / mfem_avg) << " times as long as the raw MFEM calls\n";

  EXPECT_EQ(mfem_result.Size(), expr_result.Size());
  for (int i = 0; i < rows; i++) {
    EXPECT_FLOAT_EQ(mfem_result[i], expr_result[i]);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(expr_templates, small_mixed_expr_single_alloc)
{
  MPI_Barrier(MPI_COMM_WORLD);
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

  constexpr int trials     = 100;
  double        mfem_total = 0.0;
  double        expr_total = 0.0;

  mfem::Vector mfem_result(rows);
  mfem::Vector expr_result(rows);

  // Scratchpad vectors
  mfem::Vector matvec(rows);
  mfem::Vector vec_negate_scale(rows);

  // -lhs + rhs * 3.0 - 0.3 * (matrix * vec_in)
  for (int i = 0; i < trials; i++) {
    auto mfem_start = std::chrono::steady_clock::now();
    for (int j = 0; j < trials; j++) {
      matrix.Mult(vec_in, matvec);

      add(-1.0, lhs, 3.0, rhs, vec_negate_scale);

      add(vec_negate_scale, -0.3, matvec, mfem_result);
    }
    mfem_total += (std::chrono::steady_clock::now() - mfem_start).count();
    auto expr_start = std::chrono::steady_clock::now();
    for (int j = 0; j < trials; j++) {
      evaluate(-lhs + rhs * 3.0 - 0.3 * (matrix * vec_in), expr_result);
    }
    expr_total += (std::chrono::steady_clock::now() - expr_start).count();
  }
  const auto mfem_avg = mfem_total / trials;
  const auto expr_avg = expr_total / trials;
  std::cout << "Expression templates took " << (expr_avg / mfem_avg) << " times as long as the raw MFEM calls\n";

  EXPECT_EQ(mfem_result.Size(), expr_result.Size());
  for (int i = 0; i < rows; i++) {
    EXPECT_FLOAT_EQ(mfem_result[i], expr_result[i]);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(expr_templates, large_mixed_expr)
{
  MPI_Barrier(MPI_COMM_WORLD);
  constexpr int rows = 10000;
  mfem::Vector  lhs(rows);
  mfem::Vector  rhs(rows);
  for (int i = 0; i < rows; i++) {
    lhs[i] = i * 4 + 1;
    rhs[i] = i * i * 3 + 2;
  }

  constexpr int     cols = 1200;
  mfem::Vector      vec_in(cols);
  mfem::DenseMatrix matrix(rows, cols);
  for (int i = 0; i < cols; i++) {
    vec_in[i] = i * 4 + 1;
    for (int j = 0; j < rows; j++) {
      matrix(j, i) = 2 * (i == j) - (i == (j + 1)) - (i == (j - 1));
    }
  }

  constexpr int trials     = 10;
  double        mfem_total = 0.0;
  double        expr_total = 0.0;

  mfem::Vector mfem_result(rows);
  mfem::Vector expr_result(rows);

  // -lhs + rhs * 3.0 - 0.3 * (matrix * vec_in)
  for (int i = 0; i < trials; i++) {
    auto mfem_start = std::chrono::steady_clock::now();
    for (int j = 0; j < trials; j++) {
      mfem::Vector matvec(rows);
      matrix.Mult(vec_in, matvec);

      mfem::Vector vec_negate_scale(rows);
      add(-1.0, lhs, 3.0, rhs, vec_negate_scale);

      add(vec_negate_scale, -0.3, matvec, mfem_result);
    }
    mfem_total += (std::chrono::steady_clock::now() - mfem_start).count();
    auto expr_start = std::chrono::steady_clock::now();
    for (int j = 0; j < trials; j++) {
      expr_result = -lhs + rhs * 3.0 - 0.3 * (matrix * vec_in);
    }
    expr_total += (std::chrono::steady_clock::now() - expr_start).count();
  }
  const auto mfem_avg = mfem_total / trials;
  const auto expr_avg = expr_total / trials;
  std::cout << "Expression templates took " << (expr_avg / mfem_avg) << " times as long as the raw MFEM calls\n";

  EXPECT_EQ(mfem_result.Size(), expr_result.Size());
  for (int i = 0; i < rows; i++) {
    EXPECT_FLOAT_EQ(mfem_result[i], expr_result[i]);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(expr_templates, large_mixed_expr_single_alloc)
{
  MPI_Barrier(MPI_COMM_WORLD);
  constexpr int rows = 10000;
  mfem::Vector  lhs(rows);
  mfem::Vector  rhs(rows);
  for (int i = 0; i < rows; i++) {
    lhs[i] = i * 4 + 1;
    rhs[i] = i * i * 3 + 2;
  }

  constexpr int     cols = 1200;
  mfem::Vector      vec_in(cols);
  mfem::DenseMatrix matrix(rows, cols);
  for (int i = 0; i < cols; i++) {
    vec_in[i] = i * 4 + 1;
    for (int j = 0; j < rows; j++) {
      matrix(j, i) = 2 * (i == j) - (i == (j + 1)) - (i == (j - 1));
    }
  }

  constexpr int trials     = 10;
  double        mfem_total = 0.0;
  double        expr_total = 0.0;

  mfem::Vector mfem_result(rows);
  mfem::Vector expr_result(rows);

  // Scratchpad vectors
  mfem::Vector matvec(rows);
  mfem::Vector vec_negate_scale(rows);

  // -lhs + rhs * 3.0 - 0.3 * (matrix * vec_in)
  for (int i = 0; i < trials; i++) {
    auto mfem_start = std::chrono::steady_clock::now();
    for (int j = 0; j < trials; j++) {
      matrix.Mult(vec_in, matvec);

      add(-1.0, lhs, 3.0, rhs, vec_negate_scale);

      add(vec_negate_scale, -0.3, matvec, mfem_result);
    }
    mfem_total += (std::chrono::steady_clock::now() - mfem_start).count();
    auto expr_start = std::chrono::steady_clock::now();
    for (int j = 0; j < trials; j++) {
      evaluate(-lhs + rhs * 3.0 - 0.3 * (matrix * vec_in), expr_result);
    }
    expr_total += (std::chrono::steady_clock::now() - expr_start).count();
  }
  const auto mfem_avg = mfem_total / trials;
  const auto expr_avg = expr_total / trials;
  std::cout << "Expression templates took " << (expr_avg / mfem_avg) << " times as long as the raw MFEM calls\n";

  EXPECT_EQ(mfem_result.Size(), expr_result.Size());
  for (int i = 0; i < rows; i++) {
    EXPECT_FLOAT_EQ(mfem_result[i], expr_result[i]);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(expr_templates, large_mixed_expr_single_alloc_parallel)
{
  MPI_Barrier(MPI_COMM_WORLD);
  constexpr int rows = 10000;
  mfem::Vector  lhs(rows);
  mfem::Vector  rhs(rows);
  for (int i = 0; i < rows; i++) {
    lhs[i] = i * 4 + 1;
    rhs[i] = i * i * 3 + 2;
  }

  constexpr int     cols = 1200;
  mfem::Vector      vec_in(cols);
  mfem::DenseMatrix matrix(rows, cols);
  for (int i = 0; i < cols; i++) {
    vec_in[i] = i * 4 + 1;
    for (int j = 0; j < rows; j++) {
      matrix(j, i) = 2 * (i == j) - (i == (j + 1)) - (i == (j - 1));
    }
  }

  constexpr int trials     = 10;
  double        mfem_total = 0.0;
  double        expr_total = 0.0;

  mfem::Vector mfem_result(rows);
  mfem::Vector expr_result(rows);

  // Scratchpad vectors
  mfem::Vector matvec(rows);
  mfem::Vector vec_negate_scale(rows);

  // -lhs + rhs * 3.0 - 0.3 * (matrix * vec_in)
  for (int i = 0; i < trials; i++) {
    auto mfem_start = std::chrono::steady_clock::now();
    for (int j = 0; j < trials; j++) {
      matrix.Mult(vec_in, matvec);

      add(-1.0, lhs, 3.0, rhs, vec_negate_scale);

      add(vec_negate_scale, -0.3, matvec, mfem_result);
    }
    mfem_total += (std::chrono::steady_clock::now() - mfem_start).count();
    auto expr_start = std::chrono::steady_clock::now();
    for (int j = 0; j < trials; j++) {
      evaluate(-lhs + rhs * 3.0 - 0.3 * (matrix * vec_in), expr_result, MPI_COMM_WORLD);
    }
    expr_total += (std::chrono::steady_clock::now() - expr_start).count();
  }
  const auto mfem_avg = mfem_total / trials;
  const auto expr_avg = expr_total / trials;
  std::cout << "Expression templates took " << (expr_avg / mfem_avg) << " times as long as the raw MFEM calls\n";

  EXPECT_EQ(mfem_result.Size(), expr_result.Size());
  for (int i = 0; i < rows; i++) {
    EXPECT_FLOAT_EQ(mfem_result[i], expr_result[i]);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(expr_templates, complex_expr_lambda)
{
  MPI_Barrier(MPI_COMM_WORLD);
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
  for (int i = 0; i < rows; i++) {
    EXPECT_FLOAT_EQ(mfem_result[i], expr_result[i]);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(expr_templates, small_vec_single_add_time_check)
{
  MPI_Barrier(MPI_COMM_WORLD);
  constexpr int rows = 10;
  mfem::Vector  lhs(rows);
  mfem::Vector  rhs(rows);
  for (int i = 0; i < rows; i++) {
    lhs[i] = i * 4 + 1;
    rhs[i] = i * i * 3 + 2;
  }
  constexpr int trials     = 100;
  double        mfem_total = 0.0;
  double        expr_total = 0.0;

  // Add lhs + rhs once
  for (int i = 0; i < trials; i++) {
    auto mfem_start = std::chrono::steady_clock::now();
    for (int j = 0; j < trials; j++) {
      mfem::Vector mfem_result(rows);
      add(lhs, rhs, mfem_result);
    }
    mfem_total += (std::chrono::steady_clock::now() - mfem_start).count();
    auto expr_start = std::chrono::steady_clock::now();
    for (int j = 0; j < trials; j++) {
      const mfem::Vector expr_result = lhs + rhs;
    }
    expr_total += (std::chrono::steady_clock::now() - expr_start).count();
  }
  const auto mfem_avg = mfem_total / trials;
  const auto expr_avg = expr_total / trials;
  std::cout << "Expression templates took " << (expr_avg / mfem_avg) << " times as long as the raw MFEM calls\n";
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(expr_templates, large_vec_single_add_time_check)
{
  MPI_Barrier(MPI_COMM_WORLD);
  constexpr int rows = 100000;
  mfem::Vector  lhs(rows);
  mfem::Vector  rhs(rows);
  for (int i = 0; i < rows; i++) {
    lhs[i] = i * 4 + 1;
    rhs[i] = i * i * 3 + 2;
  }
  constexpr int trials     = 10;
  double        mfem_total = 0.0;
  double        expr_total = 0.0;

  // Add lhs + rhs once
  for (int i = 0; i < trials; i++) {
    auto mfem_start = std::chrono::steady_clock::now();
    for (int j = 0; j < trials; j++) {
      mfem::Vector mfem_result(rows);
      add(lhs, rhs, mfem_result);
    }
    mfem_total += (std::chrono::steady_clock::now() - mfem_start).count();
    auto expr_start = std::chrono::steady_clock::now();
    for (int j = 0; j < trials; j++) {
      const mfem::Vector expr_result = lhs + rhs;
    }
    expr_total += (std::chrono::steady_clock::now() - expr_start).count();
  }
  const auto mfem_avg = mfem_total / trials;
  const auto expr_avg = expr_total / trials;
  std::cout << "Expression templates took " << (expr_avg / mfem_avg) << " times as long as the raw MFEM calls\n";
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(expr_templates, small_vec_multi_add_time_check)
{
  MPI_Barrier(MPI_COMM_WORLD);
  constexpr int rows = 10;
  mfem::Vector  lhs(rows);
  mfem::Vector  rhs(rows);
  for (int i = 0; i < rows; i++) {
    lhs[i] = i * 4 + 1;
    rhs[i] = i * i * 3 + 2;
  }
  constexpr int trials     = 100;
  double        mfem_total = 0.0;
  double        expr_total = 0.0;

  // Add lhs + rhs + lhs + rhs + lhs + rhs + lhs + rhs + lhs + rhs
  // Unrealistic, but aims to simulate larger expressions
  for (int i = 0; i < trials; i++) {
    auto mfem_start = std::chrono::steady_clock::now();
    for (int j = 0; j < trials; j++) {
      mfem::Vector mfem_result(rows);
      add(lhs, rhs, mfem_result);
      add(mfem_result, lhs, mfem_result);
      add(mfem_result, rhs, mfem_result);
      add(mfem_result, lhs, mfem_result);
      add(mfem_result, rhs, mfem_result);
      add(mfem_result, lhs, mfem_result);
      add(mfem_result, rhs, mfem_result);
      add(mfem_result, lhs, mfem_result);
      add(mfem_result, rhs, mfem_result);
    }
    mfem_total += (std::chrono::steady_clock::now() - mfem_start).count();
    auto expr_start = std::chrono::steady_clock::now();
    for (int j = 0; j < trials; j++) {
      const mfem::Vector expr_result = lhs + rhs + lhs + rhs + lhs + rhs + lhs + rhs + lhs + rhs;
    }
    expr_total += (std::chrono::steady_clock::now() - expr_start).count();
  }
  const auto mfem_avg = mfem_total / trials;
  const auto expr_avg = expr_total / trials;
  std::cout << "Expression templates took " << (expr_avg / mfem_avg) << " times as long as the raw MFEM calls\n";
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(expr_templates, large_vec_multi_add_time_check)
{
  MPI_Barrier(MPI_COMM_WORLD);
  constexpr int rows = 100000;
  mfem::Vector  lhs(rows);
  mfem::Vector  rhs(rows);
  for (int i = 0; i < rows; i++) {
    lhs[i] = i * 4 + 1;
    rhs[i] = i * i * 3 + 2;
  }
  constexpr int trials     = 10;
  double        mfem_total = 0.0;
  double        expr_total = 0.0;

  // Add lhs + rhs + lhs + rhs + lhs + rhs + lhs + rhs + lhs + rhs
  // Unrealistic, but aims to simulate larger expressions
  for (int i = 0; i < trials; i++) {
    auto mfem_start = std::chrono::steady_clock::now();
    for (int j = 0; j < trials; j++) {
      mfem::Vector mfem_result(rows);
      add(lhs, rhs, mfem_result);
      add(mfem_result, lhs, mfem_result);
      add(mfem_result, rhs, mfem_result);
      add(mfem_result, lhs, mfem_result);
      add(mfem_result, rhs, mfem_result);
      add(mfem_result, lhs, mfem_result);
      add(mfem_result, rhs, mfem_result);
      add(mfem_result, lhs, mfem_result);
      add(mfem_result, rhs, mfem_result);
    }
    mfem_total += (std::chrono::steady_clock::now() - mfem_start).count();
    auto expr_start = std::chrono::steady_clock::now();
    for (int j = 0; j < trials; j++) {
      const mfem::Vector expr_result = lhs + rhs + lhs + rhs + lhs + rhs + lhs + rhs + lhs + rhs;
    }
    expr_total += (std::chrono::steady_clock::now() - expr_start).count();
  }
  const auto mfem_avg = mfem_total / trials;
  const auto expr_avg = expr_total / trials;
  std::cout << "Expression templates took " << (expr_avg / mfem_avg) << " times as long as the raw MFEM calls\n";
  MPI_Barrier(MPI_COMM_WORLD);
}

//------------------------------------------------------------------------------
#include "axom/slic/core/UnitTestLogger.hpp"
using axom::slic::UnitTestLogger;

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  UnitTestLogger logger;  // create & initialize test logger, finalized when exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}

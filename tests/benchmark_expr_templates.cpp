// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <benchmark/benchmark.h>

#include <utility>

#include "numerics/expr_template_ops.hpp"

static std::pair<mfem::Vector, mfem::Vector> sample_vectors(const int entries)
{
  mfem::Vector lhs(entries);
  mfem::Vector rhs(entries);
  for (int i = 0; i < entries; i++) {
    lhs[i] = i * 4 + 1;
    rhs[i] = i * i * 3 + 2;
  }
  return {lhs, rhs};
}

static std::pair<mfem::DenseMatrix, mfem::Vector> sample_matvec(const int rows, const int cols)
{
  mfem::Vector      vec_in(cols);
  mfem::DenseMatrix matrix(rows, cols);
  for (int i = 0; i < cols; i++) {
    vec_in[i] = i * 4 + 1;
    for (int j = 0; j < rows; j++) {
      matrix(j, i) = 2 * (i == j) - (i == (j + 1)) - (i == (j - 1));
    }
  }
  return {matrix, vec_in};
}

static void BM_mixed_expr_MFEM(benchmark::State& state)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Number of rows is the argument that varies
  const int rows  = state.range(0);
  auto [lhs, rhs] = sample_vectors(rows);

  // Arbitrary
  const int cols = rows / 2;

  auto [matrix, vec_in] = sample_matvec(rows, cols);

  mfem::Vector mfem_result(rows);
  mfem::Vector expr_result(rows);

  // Scratchpad vectors
  mfem::Vector matvec(rows);
  mfem::Vector vec_negate_scale(rows);

  for (auto _ : state) {
    // This code gets timed
    matrix.Mult(vec_in, matvec);
    add(-1.0, lhs, 3.0, rhs, vec_negate_scale);
    add(vec_negate_scale, -0.3, matvec, mfem_result);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

static void BM_mixed_expr_EXPR(benchmark::State& state)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Number of rows is the argument that varies
  const int rows  = state.range(0);
  auto [lhs, rhs] = sample_vectors(rows);

  // Arbitrary
  const int cols = rows / 2;

  auto [matrix, vec_in] = sample_matvec(rows, cols);

  mfem::Vector expr_result(rows);

  for (auto _ : state) {
    // This code gets timed
    expr_result = -lhs + rhs * 3.0 - 0.3 * (matrix * vec_in);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

static void BM_mixed_expr_single_alloc_EXPR(benchmark::State& state)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Number of rows is the argument that varies
  const int rows  = state.range(0);
  auto [lhs, rhs] = sample_vectors(rows);

  // Arbitrary
  const int cols = rows / 2;

  auto [matrix, vec_in] = sample_matvec(rows, cols);

  mfem::Vector expr_result(rows);

  for (auto _ : state) {
    // This code gets timed
    evaluate(-lhs + rhs * 3.0 - 0.3 * (matrix * vec_in), expr_result);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

static void BM_large_expr_MFEM(benchmark::State& state)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Number of rows is the argument that varies
  const int rows  = state.range(0);
  auto [lhs, rhs] = sample_vectors(rows);

  mfem::Vector mfem_result(rows);

  for (auto _ : state) {
    // This code gets timed
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
    add(mfem_result, lhs, mfem_result);
    add(mfem_result, rhs, mfem_result);
    add(mfem_result, lhs, mfem_result);
    add(mfem_result, rhs, mfem_result);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

static void BM_large_expr_single_alloc_EXPR(benchmark::State& state)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Number of rows is the argument that varies
  const int rows  = state.range(0);
  auto [lhs, rhs] = sample_vectors(rows);

  mfem::Vector expr_result(rows);

  for (auto _ : state) {
    // This code gets timed
    evaluate(lhs + rhs + lhs + rhs + lhs + rhs + lhs + rhs + lhs + rhs + lhs + rhs + lhs + rhs, expr_result);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

static void BM_large_expr_single_alloc_par_EXPR(benchmark::State& state)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Number of rows is the argument that varies
  const int rows  = state.range(0);
  auto [lhs, rhs] = sample_vectors(rows);

  mfem::Vector expr_result(rows);

  for (auto _ : state) {
    // This code gets timed
    evaluate(lhs + rhs + lhs + rhs + lhs + rhs + lhs + rhs + lhs + rhs + lhs + rhs + lhs + rhs, expr_result,
             MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

BENCHMARK(BM_mixed_expr_MFEM)->RangeMultiplier(2)->Range(10, 10 << 10);
BENCHMARK(BM_mixed_expr_EXPR)->RangeMultiplier(2)->Range(10, 10 << 10);
BENCHMARK(BM_mixed_expr_single_alloc_EXPR)->RangeMultiplier(2)->Range(10, 10 << 10);
BENCHMARK(BM_large_expr_MFEM)->RangeMultiplier(2)->Range(10, 10 << 10);
BENCHMARK(BM_large_expr_single_alloc_EXPR)->RangeMultiplier(2)->Range(10, 10 << 10);
BENCHMARK(BM_large_expr_single_alloc_par_EXPR)->RangeMultiplier(2)->Range(10, 10 << 10);

//------------------------------------------------------------------------------
#include "axom/slic/core/UnitTestLogger.hpp"
using axom::slic::UnitTestLogger;

int main(int argc, char* argv[])
{
  ::benchmark::Initialize(&argc, argv);

  MPI_Init(&argc, &argv);

  UnitTestLogger logger;  // create & initialize test logger, finalized when exiting main scope

  ::benchmark::RunSpecifiedBenchmarks();

  MPI_Finalize();

  return 0;
}

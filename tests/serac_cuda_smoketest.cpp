// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <algorithm>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

void vector_add(float* out, float* a, float* b, int n);

TEST(cuda_smoketest, cuda_version)
{
  int         driverVersion  = 0;
  int         runtimeVersion = 0;
  cudaError_t error_id;

  error_id = cudaDriverGetVersion(&driverVersion);
  if (error_id != cudaSuccess) {
    std::string msg = "cudaDriverGetVersion returned CUDA Error (" + std::to_string(error_id) +
                      "): " + cudaGetErrorString(error_id) + "\n";
    FAIL() << msg;
  }
  std::cout << "CUDA driver version: " << driverVersion << std::endl;

  error_id = cudaRuntimeGetVersion(&runtimeVersion);
  if (error_id != cudaSuccess) {
    std::string msg = "cudaDriverGetVersion returned CUDA Error (" + std::to_string(error_id) +
                      "): " + cudaGetErrorString(error_id) + "\n";
    FAIL() << msg;
  }
  std::cout << "CUDA runtime version: " << runtimeVersion << std::endl;
}

TEST(cuda_smoketest, vec_add)
{
  constexpr int N = 10;

  float a[N];
  float b[N];
  float out[N];

  std::fill(a, a + N, 2.0f);
  std::fill(b, b + N, 4.0f);

  vector_add(out, a, b, N);

  std::for_each(out, out + N, [](const float f) { EXPECT_FLOAT_EQ(f, 6.0); });
}

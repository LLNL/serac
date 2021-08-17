// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file accelerator.hpp
 *
 * @brief This file contains the interface used for initializing/terminating
 * any hardware accelerator-related functionality
 */

#pragma once

#if defined(__CUDACC__)
#define SERAC_HOST_DEVICE __host__ __device__
#define SERAC_HOST __host__
#define SERAC_DEVICE __device__

/**
 * Note: nvcc will sometimes emit a warning if a __host__ __device__ function calls a __host__-only or __device__-only
 * function. make_tensor is marked __host__ __device__ and is used frequently in the code base, so it was emitting a lot
 * of warnings. This #pragma directive suppresses the warning for a specific function.
 */

#if __CUDAVER__ >= 75000
#define SERAC_SUPPRESS_NVCC_HOSTDEVICE_WARNING #pragma nv_exec_check_disable
#else
#define SERAC_SUPPRESS_NVCC_HOSTDEVICE_WARNING #pragma hd_warning_disable
#endif
#else
/**
 * @brief Macro that toggles between decorating a function for host and device or noop's for non-accelated builds.
 */
#define SERAC_HOST_DEVICE
#define SERAC_HOST
#define SERAC_DEVICE
#define SERAC_SUPPRESS_NVCC_HOSTDEVICE_WARNING

#include <cuda_runtime.h>
#endif

#include <iostream>

/**
 * @brief Accelerator functionality
 */
namespace serac {

/**
 * @brief tag type for signaling that calculations that should be performed on the CPU
 */
struct cpu_policy {
};

/**
 * @brief tag type for signaling that calculations that should be performed on the CPU
 */
struct gpu_policy {
};

/**
 * @brief for now, we'll just default to the CPU
 */
using default_policy = cpu_policy;

namespace accelerator {

/**
 * @brief Initializes the device (GPU)
 *
 * @note This function should only be called once
 */
void initializeDevice();

/**
 * @brief Cleans up the device, if applicable
 */
void terminateDevice();

#if defined(__CUDACC__)
/**
 * @brief utility method to display last cuda error message
 *
 * @param[in] o The output stream to post success or CUDA error messages
 * @param[in] success_string A string to print if there are no CUDA error messages
 */
inline void displayLastCUDAErrorMessage(std::ostream& o, const char* success_string = "")
{
  auto error = cudaGetLastError();
  if (error != cudaError::cudaSuccess) {
    o << "Last CUDA Error Message :" << cudaGetErrorString(error) << std::endl;
  } else if (strlen(success_string) > 0) {
    o << success_string << std::endl;
  }
}
#endif

}  // namespace accelerator

}  // namespace serac

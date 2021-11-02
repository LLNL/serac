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

#include <cuda_runtime.h>
#else  //__CUDACC__
/**
 * @brief Macro that evaluates to `__host__ __device__` when compiling with nvcc and does nothing on a host compiler.
 */
#define SERAC_HOST_DEVICE
/**
 * @brief Macro that evaluates to `__host__` when compiling with nvcc and does nothing on a host compiler
 */
#define SERAC_HOST
/**
 * @brief Macro that evaluates to `__device__` when compiling with nvcc and does nothing on a host compiler
 */
#define SERAC_DEVICE
/**
 * @brief Macro to turn off specific nvcc warnings
 */
#define SERAC_SUPPRESS_NVCC_HOSTDEVICE_WARNING
#endif

#include <memory>

#include "serac/infrastructure/logger.hpp"
#include "serac/infrastructure/profiling.hpp"

/**
 * @brief Accelerator functionality
 */
namespace serac {

/**
 * @brief enum used for signalling whether or not to perform certain calculations on the CPU or GPU
 */
enum class ExecutionSpace
{
  CPU,
  GPU,
  Dynamic  // Corresponds to execution that can "legally" happen on either the host or device
};

/**
 * @brief The default execution space for serac builds
 */
constexpr ExecutionSpace default_execution_space = ExecutionSpace::CPU;

/**
 * @brief Namespace for methods involving accelerator-enabled builds
 */
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
 * @brief Utility method to display last cuda error message
 *
 * @param[in] success_string A string to print if there are no CUDA error messages
 * @param[in] exit_on_error Exit on CUDA error
 */
inline void displayLastCUDAMessage(const char* success_string = "", bool exit_on_error = false)
{
  auto error = cudaGetLastError();
  if (error != cudaError::cudaSuccess) {
    if (exit_on_error) {
      SLIC_ERROR_ROOT(serac::profiling::concat("Last CUDA Error Message :", cudaGetErrorString(error)));
    } else {
      SLIC_WARNING_ROOT(serac::profiling::concat("Last CUDA Error Message :", cudaGetErrorString(error)));
    }
  } else if (strlen(success_string) > 0) {
    SLIC_INFO_ROOT(success_string);
  }
}

/**
 * @brief Utility method to query the amount of memory (bytes) that is free on the device at runtime
 *
 * Granularity appears to be about 2MB of device memory on volta.
 *
 * @return tuple of free and total memory at the moment on the device context
 */

inline std::tuple<std::size_t, std::size_t> getCUDAMemInfo()
{
  std::size_t free_memory, total_memory;
  auto        error = cudaMemGetInfo(&free_memory, &total_memory);
  displayLastCUDAMessage();
  return std::make_tuple(free_memory, total_memory);
}

/**
 * @brief returns a string with GPU memory information
 */

std::string getCUDAMemInfoString()
{
  auto [free_memory, total_memory] = getCUDAMemInfo();
  return fmt::format("Free memory: {} Total_memory: {}", free_memory, total_memory);
}

#endif

/**
 * @brief create shared_ptr to an array of `n` values of type `T`, either on the host or device
 * @tparam T the type of the value to be stored in the array
 * @tparam exec the memory space where the data lives
 * @param n how many entries to allocate in the array
 */
template <typename T, ExecutionSpace exec>
std::shared_ptr<T[]> make_shared_array(std::size_t n)
{
  if constexpr (exec == ExecutionSpace::CPU) {
    return std::shared_ptr<T[]>(new T[n]);
  }

#if defined(__CUDACC__)
  if constexpr (exec == ExecutionSpace::GPU) {
    T* data;
    cudaMalloc(&data, sizeof(T) * n);
    auto deleter = [](T* ptr) { cudaFree(ptr); };
    return std::shared_ptr<T[]>(data, deleter);
  }
#endif
}

}  // namespace accelerator

}  // namespace serac

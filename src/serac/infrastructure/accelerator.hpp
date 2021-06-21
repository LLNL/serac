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
#define SERAC_DEVICE __device__
#else
#define SERAC_HOST_DEVICE
#define SERAC_DEVICE
#endif

namespace serac::accelerator {

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

}  // namespace serac::accelerator

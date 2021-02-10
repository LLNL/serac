// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file profiling.hpp
 *
 * @brief Various helper functions and macros for profiling using Caliper
 */

#pragma once

#include <string>

#include "serac/serac_config.hpp"

#ifdef SERAC_USE_CALIPER
#include "caliper/cali-manager.h"
#include "caliper/cali.h"
#endif

/**
 * @def SERAC_MARK_FUNCTION
 * Marks a function for Caliper profiling
 */

/**
 * @def SERAC_MARK_LOOP_START(id, name)
 * Marks the beginning of a loop block for Caliper profiling
 */

/**
 * @def SERAC_MARK_LOOP_ITER(id, i)
 * Marks the beginning of a loop iteration for Caliper profiling
 */

/**
 * @def SERAC_MARK_LOOP_END(id)
 * Marks the end of a loop block for Caliper profiling
 */

#ifdef SERAC_USE_CALIPER

#define SERAC_MARK_FUNCTION CALI_CXX_MARK_FUNCTION
#define SERAC_MARK_LOOP_START(id, name) CALI_CXX_MARK_LOOP_BEGIN(id, name)
#define SERAC_MARK_LOOP_ITER(id, i) CALI_CXX_MARK_LOOP_ITERATION(id, i)
#define SERAC_MARK_LOOP_END(id) CALI_CXX_MARK_LOOP_END(id)

#else  // SERAC_USE_CALIPER not defined

// Define all these as nothing so annotated code will still compile
#define SERAC_MARK_FUNCTION
#define SERAC_MARK_LOOP_START(id, name)
#define SERAC_MARK_LOOP_ITER(id, i)
#define SERAC_MARK_LOOP_END(id)

#endif

namespace serac::profiling {
/**
 * @brief Initializes performance monitoring using the Caliper library
 * @param options The Caliper ConfigManager config string, optional
 * @see https://software.llnl.gov/Caliper/ConfigManagerAPI.html#configmanager-configuration-string-syntax
 */
void initializeCaliper(const std::string& options = "");

/**
 * @brief Concludes performance monitoring and writes collected data to a file
 */
void terminateCaliper();

}  // namespace serac::profiling

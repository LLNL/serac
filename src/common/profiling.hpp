// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file profiling.hpp
 *
 * @brief Various helper functions and macros for profiling using Caliper
 */

#ifndef PROFILING_CALIPER
#define PROFILING_CALIPER

#include <string>

#include "serac_config.hpp"

#ifdef SERAC_USE_CALIPER

#include "caliper/cali-manager.h"
#include "caliper/cali.h"

#define MARK_FUNCTION CALI_CXX_MARK_FUNCTION
#define MARK_LOOP_START(id, name) CALI_CXX_MARK_LOOP_BEGIN(id, name)
#define MARK_LOOP_ITER(id, i) CALI_CXX_MARK_LOOP_ITERATION(id, i)
#define MARK_LOOP_END(id) CALI_CXX_MARK_LOOP_END(id)

#else  // SERAC_USE_CALIPER not defined

// Define all these as nothing so annotated code will still compile
#define MARK_FUNCTION
#define MARK_LOOP_START(id, name)
#define MARK_LOOP_ITER(id, i)
#define MARK_LOOP_END(id)

#endif

namespace serac::profiling {
void initializeCaliper(const std::string& options = "runtime-report");

void terminateCaliper();
}  // namespace serac::profiling

#endif

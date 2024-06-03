// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
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
#include <sstream>

#include "serac/serac_config.hpp"

#ifdef SERAC_USE_ADIAK
#include "adiak.hpp"
#endif

#ifdef SERAC_USE_CALIPER
#include "caliper/cali-manager.h"
#include "caliper/cali.h"
#endif

#include "mpi.h"

/**
 * @def SERAC_SET_METADATA(name, data)
 * Sets metadata in adiak/caliper. Calls adiak::value
 */

#ifdef SERAC_USE_ADIAK
#define SERAC_SET_METADATA(name, data) adiak::value(name, data)
#else
#define SERAC_SET_METADATA(name, data)
#endif

/**
 * @def CALI_CXX_MARK_FUNCTION
 * No-op macro in case Serac is not built with Caliper
 */

/**
 * @def CALI_CXX_MARK_LOOP_BEGIN(id, name)
 * No-op macro in case Serac is not built with Caliper
 */

/**
 * @def CALI_CXX_MARK_LOOP_ITERATION(id, i)
 * No-op macro in case Serac is not built with Caliper
 */

/**
 * @def CALI_CXX_MARK_LOOP_END(id)
 * No-op macro in case Serac is not built with Caliper
 */

/**
 * @def CALI_MARK_BEGIN(name)
 * No-op macro in case Serac is not built with Caliper
 */

/**
 * @def CALI_MARK_END(name)
 * No-op macro in case Serac is not built with Caliper
 */

/**
 * @def CALI_CXX_MARK_SCOPE(name)
 * No-op macro in case Serac is not built with Caliper
 */

#ifndef SERAC_USE_CALIPER
#define CALI_CXX_MARK_FUNCTION
#define CALI_CXX_MARK_LOOP_BEGIN(id, name)
#define CALI_CXX_MARK_LOOP_ITERATION(id, i)
#define CALI_CXX_MARK_LOOP_END(id)
#define CALI_MARK_BEGIN(name)
#define CALI_MARK_END(name)
#define CALI_CXX_MARK_SCOPE(name)
#endif

/// profiling namespace
namespace serac::profiling {

/**
 * @brief Initializes performance monitoring using the Caliper and Adiak libraries
 * @param comm The MPI communicator (used by Adiak), optional
 * @param options The Caliper ConfigManager config string, optional
 * @see https://software.llnl.gov/Caliper/ConfigManagerAPI.html#configmanager-configuration-string-syntax
 */
void initialize([[maybe_unused]] MPI_Comm comm = MPI_COMM_WORLD, [[maybe_unused]] std::string options = "");

/**
 * @brief Concludes performance monitoring and writes collected data to a file
 */
void finalize();

/// Produces a string by applying << to all arguments
template <typename... T>
std::string concat(T... args)
{
  std::stringstream ss;
  // this fold expression is a more elegant way to implement the concatenation,
  // but nvcc incorrectly generates warning "warning: expression has no effect"
  // when using the fold expression version
  // (ss << ... << args);
  ((ss << args), ...);
  return ss.str();
}

}  // namespace serac::profiling

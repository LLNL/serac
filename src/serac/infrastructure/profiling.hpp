// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
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

/**
 * @def SERAC_MARK_START(id)
 * Marks the start of a region Caliper profiling
 */

/**
 * @def SERAC_MARK_END(id)
 * Marks the end of a region Caliper profiling
 */

/**
 * @def SERAC_PROFILE_SCOPE(name)
 * Uses cali::ScopeAnnotation to profile a particular scope
 */

/**
 * @def SERAC_PROFILE_EXPR(name, expr)
 * Profiles a single expression using a cali::ScopeAnnotation internally. Returns evaluation.
 */

/**
 * @def SERAC_PROFILE_EXPR_LOOP(name, expr, ntest)
 * Profiles an expression several times. Returns the last evaluation
 */

#ifdef SERAC_USE_ADIAK
#define SERAC_SET_METADATA(name, data) adiak::value(name, data)
#else
#define SERAC_SET_METADATA(name, data)
#endif

#ifdef SERAC_USE_CALIPER

#define SERAC_MARK_FUNCTION CALI_CXX_MARK_FUNCTION
#define SERAC_MARK_LOOP_START(id, name) CALI_CXX_MARK_LOOP_BEGIN(id, name)
#define SERAC_MARK_LOOP_ITER(id, i) CALI_CXX_MARK_LOOP_ITERATION(id, i)
#define SERAC_MARK_LOOP_END(id) CALI_CXX_MARK_LOOP_END(id)
#define SERAC_MARK_START(name) serac::profiling::detail::startCaliperRegion(name)
#define SERAC_MARK_END(name) serac::profiling::detail::endCaliperRegion(name)

#define SERAC_CONCAT_(a, b) a##b
#define SERAC_CONCAT(a, b) SERAC_CONCAT_(a, b)

namespace serac::profiling::detail {

/**
 * @brief Guarantees str is a c string
 */
inline const char* make_cstr(const char* str) { return str; }

/**
 * @brief Converts a std::string into a c string
 */
inline const char* make_cstr(const std::string& str) { return str.c_str(); }

}  // namespace serac::profiling::detail

#define SERAC_PROFILE_SCOPE(name) \
  cali::ScopeAnnotation SERAC_CONCAT(region, __LINE__)(serac::profiling::detail::make_cstr(name))

// We use decltype(auto) here instead of the default auto for a different set of type deduction rules -
// the latter uses template type deduction rules but the former uses those for decltype, which we need
// in order for the return type to take into account the value category (rvalue, lvalue) of the expression
// We have to return (expr) instead of expr to ensure that reference-ness is propagated through correctly
// in Clang - GCC handles this correctly without the parentheses as expected
#define SERAC_PROFILE_EXPR(name, expr)                                                                     \
  [&]() -> decltype(auto) {                                                                                \
    const cali::ScopeAnnotation SERAC_CONCAT(region, __LINE__)(serac::profiling::detail::make_cstr(name)); \
    return (expr);                                                                                         \
  }()

/**
 * @brief Profiles an expression several times; Return the last evaluation
 */
#define SERAC_PROFILE_EXPR_LOOP(name, expr, ntest)                                                                  \
  (                                                                                                                 \
      [&]() {                                                                                                       \
        for (int SERAC_CONCAT(i, __LINE__) = 0; SERAC_CONCAT(i, __LINE__) < ntest - 1; SERAC_CONCAT(i, __LINE__)++) \
          SERAC_PROFILE_EXPR(serac::profiling::detail::make_cstr(name), expr);                                      \
      }(),                                                                                                          \
      SERAC_PROFILE_EXPR(serac::profiling::detail::make_cstr(name), expr))

#else  // SERAC_USE_CALIPER not defined

// Define all these as nothing so annotated code will still compile
#define SERAC_MARK_FUNCTION
#define SERAC_MARK_LOOP_START(id, name)
#define SERAC_MARK_LOOP_ITER(id, i)
#define SERAC_MARK_LOOP_END(id)
#define SERAC_MARK_START(name)
#define SERAC_MARK_END(name)
#define SERAC_PROFILE_SCOPE(name)
#define SERAC_PROFILE_EXPR(name, expr) expr
#define SERAC_PROFILE_EXPR_LOOP(name, expr, ntest) expr

#endif

/// profiling namespace
namespace serac::profiling {

/**
 * @brief Initializes performance monitoring using the Caliper and Adiak libraries
 * @param comm The MPI communicator (used by Adiak), optional
 * @param options The Caliper ConfigManager config string, optional
 * @see https://software.llnl.gov/Caliper/ConfigManagerAPI.html#configmanager-configuration-string-syntax
 */
void initialize(MPI_Comm comm = MPI_COMM_WORLD, std::string options = "");

/**
 * @brief Concludes performance monitoring and writes collected data to a file
 */
void finalize();

/// detail namespace
namespace detail {

/**
 * @brief Caliper method for marking the start of a profiling region
 *
 * @param[in] name The tag to associate with the region.
 */
void startCaliperRegion(const char* name);

/**
 * @brief Caliper methods for marking the end of a region
 *
 * @param[in] name The tag to associate with the region.
 */
void endCaliperRegion(const char* name);

}  // namespace detail

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

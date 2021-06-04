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
#include <sstream>

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

/**
 * @def SERAC_MARK_START(id)
 * Marks the start of a region Caliper profiling
 */

/**
 * @def SERAC_MARK_END(id)
 * Marks the end of a region Caliper profiling
 */

/**
 * @def SERAC_SET_METADATA(name, data)
 * Sets metadata in caliper file. Calls serac::profiling::detail::setCaliperMetadata
 */

/**
 * @def SERAC_PROFILE_SCOPE(name)
 * Uses cali::ScopeAnnotation to profile a particular scope
 */

/**
 * @def SERAC_PROFILE_EXPR(name)
 * Profiles a single expression using a cali::ScopeAnnotation internally.
 */

#ifdef SERAC_USE_CALIPER

#define SERAC_MARK_FUNCTION CALI_CXX_MARK_FUNCTION
#define SERAC_MARK_LOOP_START(id, name) CALI_CXX_MARK_LOOP_BEGIN(id, name)
#define SERAC_MARK_LOOP_ITER(id, i) CALI_CXX_MARK_LOOP_ITERATION(id, i)
#define SERAC_MARK_LOOP_END(id) CALI_CXX_MARK_LOOP_END(id)
#define SERAC_MARK_START(name) serac::profiling::detail::startCaliperRegion(name)
#define SERAC_MARK_END(name) serac::profiling::detail::endCaliperRegion(name)
#define SERAC_SET_METADATA(name, data) serac::profiling::detail::setCaliperMetadata(name, data)

#define SERAC_CONCAT_(a, b) a##b
#define SERAC_CONCAT(a, b) SERAC_CONCAT_(a, b)

#define SERAC_PROFILE_SCOPE(name) cali::ScopeAnnotation SERAC_CONCAT(region, __LINE__)(name)

/// @cond
namespace detail {

/**
 * @brief Removes an rvalue reference from a type, if applicable
 * @see std::remove_reference
 * @tparam T The type to remove an rvalue reference from
 */
template <typename T>
struct remove_rvalue_reference {
  using type = T;
};

template <typename T>
struct remove_rvalue_reference<T&&> {
  using type = T;
};

/**
 * @brief Wrapper over std::forward for type deduction purposes
 * The return type participates in reference collapsing so the return type will indicate
 * the true value category of @a thing
 * @param[in] thing The object whose value category is to be deduced
 */
template <typename T>
auto&& forwarder(T&& thing)
{
  return std::forward<T>(thing);
}

}  // namespace detail
/// @endcond

/**
 * @brief The type that should be returned from the profiling wrapper lambda
 * It converts rvalue reference arguments of type T&& to T to avoid returning
 * a reference to a temporary and leaves all other types intact.
 */
#define SERAC_PROFILE_EXPR_RETURN_TYPE(expr) detail::remove_rvalue_reference<decltype(detail::forwarder(expr))>::type

#define SERAC_PROFILE_EXPR(name, expr)                                \
  [&]() -> SERAC_PROFILE_EXPR_RETURN_TYPE(expr) {                     \
    const cali::ScopeAnnotation SERAC_CONCAT(region, __LINE__)(name); \
    return static_cast<SERAC_PROFILE_EXPR_RETURN_TYPE(expr)>(expr);   \
  }()

#else  // SERAC_USE_CALIPER not defined

// Define all these as nothing so annotated code will still compile
#define SERAC_MARK_FUNCTION
#define SERAC_MARK_LOOP_START(id, name)
#define SERAC_MARK_LOOP_ITER(id, i)
#define SERAC_MARK_LOOP_END(id)
#define SERAC_MARK_START(name)
#define SERAC_MARK_END(name)
#define SERAC_SET_METADATA(name, data)
#define SERAC_PROFILE_SCOPE(name)
#define SERAC_PROFILE_EXPR(name, expr) expr

#endif

/// profiling namespace
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

/// detail namespace
namespace detail {
/**
 * @brief Caliper metadata methods cali_set_global_<double|int|string|uint>_byname()
 *
 * @param[in] name The tag to associate the following metadata with
 * @param[in] data The metadata to store in the caliper file
 */
void setCaliperMetadata(const std::string& name, const std::string& data);

/*!
  @overload
*/
void setCaliperMetadata(const std::string& name, int data);

/*!
  @overload
*/
void setCaliperMetadata(const std::string& name, double data);

/*!
  @overload
*/
void setCaliperMetadata(const std::string& name, unsigned int data);

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

/// @brief Provides the member typedef type that names T.
/// see: https://en.cppreference.com/w/cpp/types/type_identity
template <class T>
struct type_identity {
  /// @brief The member typedef type that names T.
  using type = T;
};

/// @brief Helper type for type identity
template <class T>
using type_identity_t = typename type_identity<T>::type;

}  // namespace detail

/// Produces a string by applying << to all arguments
template <typename... T>
std::string concat(T... args)
{
  std::stringstream ss;
  (ss << ... << args);
  return ss.str();
}

}  // namespace serac::profiling

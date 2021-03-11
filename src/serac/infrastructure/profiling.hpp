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

#ifdef SERAC_USE_CALIPER

#define SERAC_MARK_FUNCTION CALI_CXX_MARK_FUNCTION
#define SERAC_MARK_LOOP_START(id, name) CALI_CXX_MARK_LOOP_BEGIN(id, name)
#define SERAC_MARK_LOOP_ITER(id, i) CALI_CXX_MARK_LOOP_ITERATION(id, i)
#define SERAC_MARK_LOOP_END(id) CALI_CXX_MARK_LOOP_END(id)
#define SERAC_MARK_START(name) CALI_MARK_BEGIN(name)
#define SERAC_MARK_END(name) CALI_MARK_END(name)
#define SERAC_SET_METADATA(name, data) serac::profiling::detail::setCaliperMetadata(name, data)

#else  // SERAC_USE_CALIPER not defined

// Define all these as nothing so annotated code will still compile
#define SERAC_MARK_FUNCTION
#define SERAC_MARK_LOOP_START(id, name)
#define SERAC_MARK_LOOP_ITER(id, i)
#define SERAC_MARK_LOOP_END(id)
#define SERAC_MARK_START(name)
#define SERAC_MARK_END(name)
#define SERAC_SET_METADATA(name, data) 

#endif

// profiling namespace
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
     * @fn void serac::profiling::setCaliperMetadata(const std::string & name, double data)
     * 
     * Caliper metadata methods cali_set_global_(double|int|string|uint)_byname(
     */
    void setCaliperMetadata(const std::string& name, double data);

    /*! 
      @overload void serac::profiling::setCaliperMetadata(const std::string & name, int data)
    */
    void setCaliperMetadata(const std::string& name, int data);

    /*! 
      @overload void serac::profiling::setCaliperMetadata(const std::string & name, std::string data)
    */
    void setCaliperMetadata(const std::string& name, std::string data);

    /*! 
      @overload void serac::profiling::setCaliperMetadata(const std::string & name, unsigned int data)
    */
    void setCaliperMetadata(const std::string& name, unsigned int data);
  } // namespace serac::profiling::detail
  
  
}  // namespace serac::profiling

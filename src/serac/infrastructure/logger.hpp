// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file logger.hpp
 *
 * @brief This file contains the all the necessary functions and macros required
 *        for logging as well as a helper function to exit the program gracefully.
 */

#pragma once

#include "axom/slic.hpp"
#include "fmt/fmt.hpp"
#include "mpi.h"

// Logger functionality
namespace serac::logger {
/**
 * @brief Initializes and setups the logger.
 *
 * Setups and tailors the SLIC logger for Serac.  Sets the SLIC loggings streams
 * and tells SLIC how to format the messages.  This function also creates different
 * logging streams if you are running serial, parallel, or parallel with Lumberjack
 * support.
 *
 * @param[in] comm MPI communicator that the logger will use
 */
bool initialize(MPI_Comm comm);

/**
 * @brief Finalizes the logger.
 *
 * Closes and finalizes the SLIC logger.
 */
void finalize();

/**
 * @brief Flushes messages currently held by the logger.
 *
 * If running in parallel, SLIC doesn't output messages immediately.  This flushes
 * all messages currently held by SLIC.  This is a collective operation because
 * messages can be spread across MPI ranks.
 */
void flush();

}  // namespace serac::logger

// Utility SLIC macros

/**
 * @brief Macro that logs given error message only on rank 0.
 */
#define SLIC_ERROR_ROOT(rank, msg) SLIC_ERROR_IF(rank == 0, msg)

/**
 * @brief Macro that logs given warning message only on rank 0.
 */
#define SLIC_WARNING_ROOT(rank, msg) SLIC_WARNING_IF(rank == 0, msg)

/**
 * @brief Macro that logs given info message only on rank 0.
 */
#define SLIC_INFO_ROOT(rank, msg) SLIC_INFO_IF(rank == 0, msg)

/**
 * @brief Macro that logs given debug message only on rank 0.
 */
#define SLIC_DEBUG_ROOT(rank, msg) SLIC_DEBUG_IF(rank == 0, msg)

/**
 * @brief Macro that logs given error message only on rank 0 if EXP is true.
 */
#define SLIC_ERROR_ROOT_IF(EXP, rank, msg) SLIC_ERROR_IF((EXP) && (rank == 0), msg)

/**
 * @brief Macro that logs given warning message only on rank 0 if EXP is true.
 */
#define SLIC_WARNING_ROOT_IF(EXP, rank, msg) SLIC_WARNING_IF((EXP) && (rank == 0), msg)

/**
 * @brief Macro that logs given info message only on rank 0 if EXP is true.
 */
#define SLIC_INFO_ROOT_IF(EXP, rank, msg) SLIC_INFO_IF((EXP) && (rank == 0), msg)

/**
 * @brief Macro that logs given debug message only on rank 0 if EXP is true.
 */
#define SLIC_DEBUG_ROOT_IF(EXP, rank, msg) SLIC_DEBUG_IF((EXP) && (rank == 0), msg)

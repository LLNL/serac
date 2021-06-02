// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file terminator.hpp
 *
 * @brief Helper functions for exiting Serac cleanly
 */

#pragma once

namespace serac {

/**
 * @brief Terminator functionality
 */
namespace terminator {

/**
 * Registers the signalHandler function to handle various fatal
 * signals
 * @note The behavior of MPI when a signal is sent to the mpirun
 * process is implementation-defined.  OpenMPI will send a SIGTERM
 * (which can be caught) and then a SIGKILL (which cannot) a few
 * seconds later.  PlatformMPI will propagate the signal to the
 * individual processes.  Before propagating or sending a signal,
 * MPI may or may not call MPI_Finalize/MPI_Abort.
 */
void registerSignals();

}  // namespace terminator

/**
 * @brief Exits the program gracefully after cleaning up necessary tasks.
 *
 * This performs finalization work needed by the program such as finalizing MPI
 * and flushing and closing the SLIC logger.
 *
 * @param[in] error True if the program should return an error code
 */
void exitGracefully(bool error = false);

}  // namespace serac

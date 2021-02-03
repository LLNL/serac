// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file initialize.hpp
 *
 * @brief A function intended to be used as part of a driver to initialize common libraries
 */
#pragma once

#include <utility>

#include "mpi.h"

namespace serac {

/**
 * @brief Returns the number of processes and rank for an MPI communicator
 * @param comm The MPI communicator to initialize with
 * @return A pair containing {size, rank} relative to the provided MPI communicator
 * @pre The logger must be initialized (to display error messages)
 */
std::pair<int, int> getMPIInfo(MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Initializes MPI, signal handling, and logging
 * @param argc The number of command-line arguments
 * @param argv The command-line arguments, as C-strings
 * @param comm The MPI communicator to initialize with
 * @return A pair containing the size and rank relative to the provided MPI communicator
 */
std::pair<int, int> initialize(int argc, char* argv[], MPI_Comm comm = MPI_COMM_WORLD);

}  // namespace serac

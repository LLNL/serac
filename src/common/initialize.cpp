// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "common/initialize.hpp"

#include "common/logger.hpp"
#include "common/profiling.hpp"
#include "common/terminator.hpp"

namespace serac {

std::pair<int, int> getMPIInfo(MPI_Comm comm)
{
  int num_procs = 0;
  int rank      = 0;
  if (MPI_Comm_size(comm, &num_procs) != MPI_SUCCESS) {
    serac::exitGracefully(true);
  }

  if (MPI_Comm_rank(comm, &rank) != MPI_SUCCESS) {
    serac::exitGracefully(true);
  }
  return {num_procs, rank};
}

std::pair<int, int> initialize(int argc, char* argv[], MPI_Comm comm)
{
  // Initialize MPI.
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    serac::exitGracefully(true);
  }

  // Initialize the signal handler
  terminator::registerSignals();

  // Initialize SLIC logger
  if (!logger::initialize(comm)) {
    serac::exitGracefully(true);
  }

  // Start the profiler (no-op if not enabled)
  profiling::initializeCaliper();

  return getMPIInfo(comm);
}

}  // namespace serac

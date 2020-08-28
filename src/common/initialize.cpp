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

std::pair<int, int> initialize(int argc, char* argv[], MPI_Comm comm)
{
  // Initialize MPI.
  int num_procs, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(comm, &num_procs);
  MPI_Comm_rank(comm, &rank);

  // Initialize the signal handler
  terminator::registerSignals();

  // Initialize SLIC logger
  if (!logger::initialize(comm)) {
    serac::exitGracefully(true);
  }

  // Start the profiler (no-op if not enabled)
  profiling::initializeCaliper();

  return {num_procs, rank};
}

}  // namespace serac

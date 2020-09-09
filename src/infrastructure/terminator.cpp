// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "infrastructure/terminator.hpp"

#include <csignal>
#include <cstdlib>
#include <iostream>

#include "infrastructure/logger.hpp"
#include "infrastructure/profiling.hpp"

namespace {
/**
 * The actual signal handler - must have this exact signature
 * MPI is probably dead by this point, SLIC is definitely dead
 */
void signalHandler(int signal)
{
  std::cerr << "[SIGNAL]: Received signal " << signal << ", exiting" << std::endl;
  serac::exitGracefully(true);
}

}  // namespace

namespace serac {

namespace terminator {

void registerSignals()
{
  std::signal(SIGINT, signalHandler);
  std::signal(SIGABRT, signalHandler);
  std::signal(SIGSEGV, signalHandler);
  std::signal(SIGTERM, signalHandler);
}

}  // namespace terminator

void exitGracefully(bool error)
{
  if (axom::slic::isInitialized()) {
    serac::logger::flush();
    serac::logger::finalize();
  }
  int mpi_finalized;
  MPI_Finalized(&mpi_finalized);
  if (!mpi_finalized) {
    MPI_Finalize();
  }
  profiling::terminateCaliper();
  error ? exit(EXIT_FAILURE) : exit(EXIT_SUCCESS);
}

}  // namespace serac

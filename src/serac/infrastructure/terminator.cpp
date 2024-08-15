// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/infrastructure/terminator.hpp"

#include <csignal>
#include <cstdlib>
#include <iostream>

#include "serac/infrastructure/accelerator.hpp"
#include "serac/infrastructure/logger.hpp"
#include "serac/infrastructure/profiling.hpp"

#include "mfem.hpp"

namespace {
/**
 * The actual signal handler - must have this exact signature
 * MPI is probably dead by this point, SLIC is definitely dead
 */
void signalHandler(int signal)
{
  std::cerr << "[SIGNAL]: Received signal " << signal << " (" << strsignal(signal) << "), exiting" << std::endl;
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

#ifdef SERAC_USE_PETSC
#ifdef SERAC_USE_SLEPC
  mfem::MFEMFinalizeSlepc();
#else
  mfem::MFEMFinalizePetsc();
#endif
#endif

  profiling::finalize();

  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  int mpi_finalized = 0;
  MPI_Finalized(&mpi_finalized);
  if (mpi_initialized && !mpi_finalized) {
    MPI_Finalize();
  }

  accelerator::terminateDevice();

  error ? exit(EXIT_FAILURE) : exit(EXIT_SUCCESS);
}

}  // namespace serac

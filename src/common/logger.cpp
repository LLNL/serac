// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "common/logger.hpp"

#include <csignal>
#include <cstdlib>

#include "axom/slic.hpp"
#include "mpi.h"

namespace {
/**
 * The actual signal handler - must have this exact signature
 * MPI is probably dead by this point, SLIC is definitely dead
 */
void signalHandler(int signal)
{
  int mpi_finalized;
  MPI_Finalized(&mpi_finalized);
  if (!mpi_finalized) {
    MPI_Finalize();
  }
  // This works, but is it UB to call MPI_Comm_rank here?
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::cerr << "[SIGNAL]: Received signal " << signal << ", exiting" << std::endl;
  }
  exit(EXIT_FAILURE);
}

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
void registerSignals()
{
  std::signal(SIGINT, signalHandler);
  std::signal(SIGABRT, signalHandler);
  std::signal(SIGSEGV, signalHandler);
  std::signal(SIGTERM, signalHandler);
}
}  // namespace

namespace serac {

void exitGracefully(bool error)
{
  serac::logger::flush();
  serac::logger::finalize();
  MPI_Finalize();
  error ? exit(EXIT_FAILURE) : exit(EXIT_SUCCESS);
}

namespace logger {

bool initialize(MPI_Comm comm)
{
  namespace slic = axom::slic;

  if (!slic::isInitialized()) {
    slic::initialize();
  }

  registerSignals();

  int numRanks, rank;
  MPI_Comm_size(comm, &numRanks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string loggerName = numRanks > 1 ? "serac_parallel_logger" : "serac_serial_logger";
  slic::createLogger(loggerName);
  slic::activateLogger(loggerName);
  if (!slic::activateLogger(loggerName)) {
    // Can't log through SLIC since it just failed to activate
    std::cerr << "Error: Failed to activate logger: " << loggerName << std::endl;
    return false;
  }

  // separate streams: warnings and errors (we); info and debug (id)
  slic::LogStream* idStream;
  slic::LogStream* weStream;

  std::string fmt_id = "[<LEVEL>]: <MESSAGE>\n";
  std::string fmt_we = "[<LEVEL> (<FILE>:<LINE>)]\n<MESSAGE>\n\n";

  // Only create a parallel logger when there is more than one rank
  if (numRanks > 1) {
    fmt_id = "[<RANK>]" + fmt_id;
    fmt_we = "[<RANK>]" + fmt_we;

#ifdef SERAC_USE_LUMBERJACK
    const int RLIMIT = 8;
    idStream         = new slic::LumberjackStream(&std::cout, comm, RLIMIT, fmt_id);
    weStream         = new slic::LumberjackStream(&std::cerr, comm, RLIMIT, fmt_we);
#else
    idStream = new slic::SynchronizedStream(&std::cout, comm, fmt_id);
    weStream = new slic::SynchronizedStream(&std::cerr, comm, fmt_we);
#endif
  } else {
    idStream = new slic::GenericOutputStream(&std::cout, fmt_id);
    weStream = new slic::GenericOutputStream(&std::cerr, fmt_we);
  }

  slic::setLoggingMsgLevel(slic::message::Debug);

  addStreamToMsgLevel(weStream, slic::message::Error);
  addStreamToMsgLevel(weStream, slic::message::Warning);
  addStreamToMsgLevel(idStream, slic::message::Info);
  addStreamToMsgLevel(idStream, slic::message::Debug);

  slic::setAbortOnError(false);
  slic::setAbortOnWarning(false);

  std::string msg = fmt::format("Logger activated: {0}", loggerName);
  SLIC_INFO_ROOT(rank, msg);
  serac::logger::flush();

  return true;
}

void finalize() { axom::slic::finalize(); }

void flush() { axom::slic::flushStreams(); }

}  // namespace logger
}  // namespace serac

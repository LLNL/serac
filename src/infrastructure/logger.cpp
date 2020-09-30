// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "infrastructure/logger.hpp"

namespace serac {

namespace logger {

bool initialize(MPI_Comm comm)
{
  namespace slic = axom::slic;

  if (!slic::isInitialized()) {
    slic::initialize();
  }

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

  // Separate streams for different message levels
  slic::LogStream* iStream;   // info
  slic::LogStream* dStream;   // debug
  slic::LogStream* weStream;  // warnings and errors

  std::string fmt_i  = "<MESSAGE>\n";
  std::string fmt_d  = "[<LEVEL>]: <MESSAGE>\n";
  std::string fmt_we = "[<LEVEL> (<FILE>:<LINE>)]\n<MESSAGE>\n\n";

  // Only create a parallel logger when there is more than one rank
  if (numRanks > 1) {
    fmt_i  = "[<RANK>] " + fmt_i;
    fmt_d  = "[<RANK>]" + fmt_d;
    fmt_we = "[<RANK>]" + fmt_we;

#ifdef SERAC_USE_LUMBERJACK
    const int RLIMIT = 8;
    iStream          = new slic::LumberjackStream(&std::cout, comm, RLIMIT, fmt_i);
    dStream          = new slic::LumberjackStream(&std::cout, comm, RLIMIT, fmt_d);
    weStream         = new slic::LumberjackStream(&std::cerr, comm, RLIMIT, fmt_we);
#else
    iStream  = new slic::SynchronizedStream(&std::cout, comm, fmt_i);
    dStream  = new slic::SynchronizedStream(&std::cout, comm, fmt_d);
    weStream = new slic::SynchronizedStream(&std::cerr, comm, fmt_we);
#endif
  } else {
    iStream  = new slic::GenericOutputStream(&std::cout, fmt_i);
    dStream  = new slic::GenericOutputStream(&std::cout, fmt_d);
    weStream = new slic::GenericOutputStream(&std::cerr, fmt_we);
  }

  slic::setLoggingMsgLevel(slic::message::Debug);

  addStreamToMsgLevel(weStream, slic::message::Error);
  addStreamToMsgLevel(weStream, slic::message::Warning);
  addStreamToMsgLevel(iStream, slic::message::Info);
  addStreamToMsgLevel(dStream, slic::message::Debug);

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

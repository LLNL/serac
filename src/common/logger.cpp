// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "common/logger.hpp"

#include "axom/slic.hpp"

#include <cstdlib>
#include "mpi.h"

namespace serac {

void ExitGracefully(bool error)
{
    serac::logger::Flush();
    serac::logger::Finalize();
    MPI_Finalize();
    error ? exit(EXIT_FAILURE) : exit(EXIT_SUCCESS);
}

namespace logger {

bool Initialize(MPI_Comm comm)
{
  namespace slic = axom::slic;

  if ( ! slic::isInitialized() )
  {
    slic::initialize();
  }

  int numRanks, rank;
  MPI_Comm_size(comm, &numRanks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string loggerName = numRanks > 1 ? "serac_parallel_logger" : "serac_serial_logger";
  slic::createLogger(loggerName);
  slic::activateLogger(loggerName);
  if (!slic::activateLogger(loggerName))
  {
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
  if( numRanks > 1 )
  {
    fmt_id = "[<RANK>]" + fmt_id;
    fmt_we = "[<RANK>]" + fmt_we;

    #ifdef SERAC_USE_LUMBERJACK
      const int RLIMIT = 8;
      idStream = new slic::LumberjackStream(&std::cout, comm, RLIMIT, fmt_id);
      weStream = new slic::LumberjackStream(&std::cerr, comm, RLIMIT, fmt_we);
    #else
      idStream = new slic::SynchronizedStream(&std::cout, comm, fmt_id);
      weStream = new slic::SynchronizedStream(&std::cerr, comm, fmt_we);
    #endif
  }  
  else
  {
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
  SLIC_INFO_RANK0(rank, msg);
  serac::logger::Flush();

  return true;
}

void Finalize()
{
  axom::slic::finalize();
}

void Flush()
{
  axom::slic::flushStreams();
}

} // namespace logger
} // namespace serac

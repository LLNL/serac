// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/infrastructure/logger.hpp"

#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/terminator.hpp"

#include <fstream>

namespace serac::logger {

static int logger_rank = 0;

int rank() { return logger_rank; }

bool initialize(MPI_Comm comm)
{
  namespace slic = axom::slic;

  if (!slic::isInitialized()) {
    slic::initialize();
  }

  auto [num_ranks, rank] = getMPIInfo(comm);
  logger_rank            = rank;

  std::string loggerName = num_ranks > 1 ? "serac_parallel_logger" : "serac_serial_logger";
  slic::createLogger(loggerName);
  slic::activateLogger(loggerName);
  if (!slic::activateLogger(loggerName)) {
    // Can't log through SLIC since it just failed to activate
    std::cerr << "Error: Failed to activate logger: " << loggerName << std::endl;
    return false;
  }

  // Console streams, std::cout for info/debug, std::cerr for warnings/errors
  slic::LogStream* i_logstream  = nullptr;  // info
  slic::LogStream* d_logstream  = nullptr;  // debug
  slic::LogStream* we_logstream = nullptr;  // warnings and errors

  // Stream formatting strings
  std::string i_format_string  = "<MESSAGE>\n";
  std::string d_format_string  = "[<LEVEL>]: <MESSAGE>\n";
  std::string we_format_string = "[<LEVEL> (<FILE>:<LINE>)]\n<MESSAGE>\n\n";

  // Only create a parallel logger when there is more than one rank
  if (num_ranks > 1) {
    // Add rank to format strings if parallel
    // Note: i_format_string's extra space is on purpose due to no space on above string
    i_format_string  = "[<RANK>] " + i_format_string;
    d_format_string  = "[<RANK>]" + d_format_string;
    we_format_string = "[<RANK>]" + we_format_string;

#ifdef SERAC_USE_LUMBERJACK
    const int RLIMIT = 8;

    i_logstream  = new slic::LumberjackStream(&std::cout, comm, RLIMIT, i_format_string);
    d_logstream  = new slic::LumberjackStream(&std::cout, comm, RLIMIT, d_format_string);
    we_logstream = new slic::LumberjackStream(&std::cerr, comm, RLIMIT, we_format_string);
#else
    i_logstream  = new slic::SynchronizedStream(&std::cout, comm, i_format_string);
    d_logstream  = new slic::SynchronizedStream(&std::cout, comm, d_format_string);
    we_logstream = new slic::SynchronizedStream(&std::cerr, comm, we_format_string);
#endif
  } else {
    i_logstream  = new slic::GenericOutputStream(&std::cout, i_format_string);
    d_logstream  = new slic::GenericOutputStream(&std::cout, d_format_string);
    we_logstream = new slic::GenericOutputStream(&std::cerr, we_format_string);
  }

  slic::setLoggingMsgLevel(slic::message::Debug);

  // Add message levels to streams
  addStreamToMsgLevel(i_logstream, slic::message::Info);
  addStreamToMsgLevel(d_logstream, slic::message::Debug);
  addStreamToMsgLevel(we_logstream, slic::message::Warning);
  addStreamToMsgLevel(we_logstream, slic::message::Error);

  // Exit gracefully on error
  slic::setAbortFunction([]() { exitGracefully(true); });
  slic::setAbortOnError(true);
  slic::setAbortOnWarning(false);

  std::string msg = axom::fmt::format("Logger activated: {0}", loggerName);
  SLIC_INFO_ROOT(msg);
  serac::logger::flush();

  return true;
}

void finalize() { axom::slic::finalize(); }

void flush() { axom::slic::flushStreams(); }

}  // namespace serac::logger

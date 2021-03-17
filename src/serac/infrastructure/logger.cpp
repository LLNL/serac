// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
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

// output file stream for SLIC file logger stream
static std::unique_ptr<std::ostream> logger_file_stream;

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
  slic::LogStream* i_console_stream  = nullptr;  // info
  slic::LogStream* d_console_stream  = nullptr;  // debug
  slic::LogStream* we_console_stream = nullptr;  // warnings and errors

  // File streams, all message levels go to one file
  if (rank == 0) {
    // Only root node writes/opens the file
    auto file_stream = std::make_unique<std::ofstream>();
    file_stream->open("serac.out", std::ofstream::out);
    logger_file_stream = std::move(file_stream);
  } else {
    // Create a noop stream for non-root nodes since they won't write to them anyways
    logger_file_stream = std::make_unique<std::ostream>(nullptr);
  }

  slic::LogStream* i_file_stream  = nullptr;
  slic::LogStream* d_file_stream  = nullptr;
  slic::LogStream* we_file_stream = nullptr;

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

    // Console streams
    i_console_stream  = new slic::LumberjackStream(&std::cout, comm, RLIMIT, i_format_string);
    d_console_stream  = new slic::LumberjackStream(&std::cout, comm, RLIMIT, d_format_string);
    we_console_stream = new slic::LumberjackStream(&std::cerr, comm, RLIMIT, we_format_string);

    // File streams
    i_file_stream  = new slic::LumberjackStream(logger_file_stream.get(), comm, RLIMIT, i_format_string);
    d_file_stream  = new slic::LumberjackStream(logger_file_stream.get(), comm, RLIMIT, d_format_string);
    we_file_stream = new slic::LumberjackStream(logger_file_stream.get(), comm, RLIMIT, we_format_string);
#else
    // Console streams
    i_console_stream  = new slic::SynchronizedStream(&std::cout, comm, i_format_string);
    d_console_stream  = new slic::SynchronizedStream(&std::cout, comm, d_format_string);
    we_console_stream = new slic::SynchronizedStream(&std::cerr, comm, we_format_string);

    // File streams
    i_file_stream  = new slic::SynchronizedStream(logger_file_stream.get(), comm, i_format_string);
    d_file_stream  = new slic::SynchronizedStream(logger_file_stream.get(), comm, d_format_string);
    we_file_stream = new slic::SynchronizedStream(logger_file_stream.get(), comm, we_format_string);
#endif
  } else {
    // Console streams
    i_console_stream  = new slic::GenericOutputStream(&std::cout, i_format_string);
    d_console_stream  = new slic::GenericOutputStream(&std::cout, d_format_string);
    we_console_stream = new slic::GenericOutputStream(&std::cerr, we_format_string);

    // File streams
    i_file_stream  = new slic::GenericOutputStream(logger_file_stream.get(), i_format_string);
    d_file_stream  = new slic::GenericOutputStream(logger_file_stream.get(), d_format_string);
    we_file_stream = new slic::GenericOutputStream(logger_file_stream.get(), we_format_string);
  }

  slic::setLoggingMsgLevel(slic::message::Debug);

  // Add message levels to console streams
  addStreamToMsgLevel(i_console_stream, slic::message::Info);
  addStreamToMsgLevel(d_console_stream, slic::message::Debug);
  addStreamToMsgLevel(we_console_stream, slic::message::Warning);
  addStreamToMsgLevel(we_console_stream, slic::message::Error);

  // Add message levels to file streams
  addStreamToMsgLevel(i_file_stream, slic::message::Info);
  addStreamToMsgLevel(d_file_stream, slic::message::Debug);
  addStreamToMsgLevel(we_file_stream, slic::message::Warning);
  addStreamToMsgLevel(we_file_stream, slic::message::Error);

  // Exit gracefully on error
  slic::setAbortFunction([]() { exitGracefully(true); });
  slic::setAbortOnError(true);
  slic::setAbortOnWarning(false);

  std::string msg = fmt::format("Logger activated: {0}", loggerName);
  SLIC_INFO_ROOT(msg);
  serac::logger::flush();

  return true;
}

void finalize() { axom::slic::finalize(); }

void flush() { axom::slic::flushStreams(); }

}  // namespace serac::logger

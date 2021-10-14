// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/infrastructure/initialize.hpp"

#ifdef WIN32
#include <windows.h>
#include <tchar.h>
#else
#include <unistd.h>
#include <limits.h>
#endif

#include "serac/infrastructure/accelerator.hpp"
#include "serac/infrastructure/logger.hpp"
#include "serac/infrastructure/profiling.hpp"
#include "serac/infrastructure/terminator.hpp"

namespace serac {

std::pair<int, int> getMPIInfo(MPI_Comm comm)
{
  int num_procs = 0;
  int rank      = 0;
  if (MPI_Comm_size(comm, &num_procs) != MPI_SUCCESS) {
    SLIC_ERROR("Failed to determine number of MPI processes");
  }

  if (MPI_Comm_rank(comm, &rank) != MPI_SUCCESS) {
    SLIC_ERROR("Failed to determine MPI rank");
  }
  return {num_procs, rank};
}

#define INFO_BUFFER_SIZE 32767

std::string getHostName()
{
  std::string hostName = "";
#ifdef WIN32
  TCHAR infoBuf[INFO_BUFFER_SIZE];
  DWORD bufCharCount = INFO_BUFFER_SIZE;
  bufCharCount       = INFO_BUFFER_SIZE;
  if (!GetComputerName(infoBuf, &bufCharCount)) {
    hostName = std::string(infoBuf);
  }
#else
  char infoBuf[HOST_NAME_MAX];
  if (gethostname(infoBuf, HOST_NAME_MAX) == 0) {
    hostName = std::string(infoBuf);
  }
#endif
  return hostName;
}

std::string getUserName()
{
  std::string userName = "";
#ifdef WIN32
  TCHAR infoBuf[INFO_BUFFER_SIZE];
  DWORD bufCharCount = INFO_BUFFER_SIZE;
  bufCharCount       = INFO_BUFFER_SIZE;
  if (GetUserName(infoBuf, &bufCharCount)) {
    userName = std::string(infoBuf);
  }
#else
  char infoBuf[LOGIN_NAME_MAX];
  if (getlogin_r(infoBuf, LOGIN_NAME_MAX) == 0) {
    userName = std::string(infoBuf);
  }
#endif
  return userName;
}

std::pair<int, int> initialize(int argc, char* argv[], MPI_Comm comm)
{
  // Initialize MPI.
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cerr << "Failed to initialize MPI" << std::endl;
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

  // Initialize GPU (no-op if not enabled/available)
  // TODO for some reason this causes errors on Lassen. We need to look into this ASAP.
  // accelerator::initializeDevice();

  return getMPIInfo(comm);
}

}  // namespace serac

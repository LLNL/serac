// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "infrastructure/initialize.hpp"

#include "infrastructure/logger.hpp"
#include "infrastructure/profiling.hpp"
#include "infrastructure/terminator.hpp"
#include "mfem.hpp"

namespace serac {

#ifdef MFEM_USE_CUDA
// Keep as a global so it has the same
// lifetime as the program
static mfem::Device device;
#endif

std::pair<int, int> getMPIInfo(MPI_Comm comm)
{
  int num_procs = 0;
  int rank      = 0;
  if (MPI_Comm_size(comm, &num_procs) != MPI_SUCCESS) {
    SLIC_ERROR("Failed to determine number of MPI processes");
    serac::exitGracefully(true);
  }

  if (MPI_Comm_rank(comm, &rank) != MPI_SUCCESS) {
    SLIC_ERROR("Failed to determine MPI rank");
    serac::exitGracefully(true);
  }
  return {num_procs, rank};
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

  const auto mpi_info = getMPIInfo(comm);

// If MFEM supports CUDA, configure it
#ifdef MFEM_USE_CUDA
  device.Configure("cuda");
  if (mpi_info.second == 0) {
    device.Print();
  }
#endif

  return mpi_info;
}

}  // namespace serac

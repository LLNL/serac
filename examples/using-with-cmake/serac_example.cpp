// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file serac.cpp
 *
 * @brief Basic serac example
 *
 * Intended to verify that external projects can include Serac
 */

#include "infrastructure/initialize.hpp"
#include "infrastructure/logger.hpp"
#include "infrastructure/terminator.hpp"
#include "mfem.hpp"
#include "axom/core.hpp"

int main(int argc, char* argv[])
{
  auto [num_procs, rank] = serac::initialize(argc, argv);

  SLIC_INFO_ROOT(rank, "MFEM version: " << mfem::GetVersionStr());

  axom::about();

  SLIC_INFO_ROOT(rank, "Serac loaded successfully.");

  serac::exitGracefully();
}

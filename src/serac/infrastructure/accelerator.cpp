// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/infrastructure/accelerator.hpp"

#include <memory>

#include "serac/infrastructure/logger.hpp"

namespace serac {

namespace accelerator {

// Restrict global to this file only
namespace {
std::unique_ptr<mfem::Device> device;
}  // namespace

void initializeDevice(ExecutionSpace exec)
{
  SLIC_ERROR_ROOT_IF(device, "serac::accelerator::initializeDevice cannot be called more than once");
  device = std::make_unique<mfem::Device>();
  if (exec == ExecutionSpace::GPU) {
#if defined(MFEM_USE_CUDA)
    device->Configure("cuda");
#endif
  }
}

void terminateDevice(ExecutionSpace exec)
{
  // Idempotent, no adverse affects if called multiple times
  if (exec == ExecutionSpace::GPU) {
    device.reset();
  }
}

}  // namespace accelerator

}  // namespace serac

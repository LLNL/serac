// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/infrastructure/accelerator.hpp"

#include <memory>

#include "mfem.hpp"

#include "serac/infrastructure/logger.hpp"

namespace serac {

namespace accelerator {

// Restrict global to this file only
namespace {
std::unique_ptr<mfem::Device> device;
}  // namespace

void initializeDevice()
{
  SLIC_ERROR_ROOT_IF(device, "serac::accelerator::initializeDevice cannot be called more than once");
  device = std::make_unique<mfem::Device>();
#ifdef MFEM_USE_CUDA
  device->Configure("cuda");
#endif
}

void terminateDevice()
{
  // Idempotent, no adverse affects if called multiple times
  device.reset();
}

}  // namespace accelerator

}  // namespace serac

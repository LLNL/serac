// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file mfem_configs.hpp
 *
 * @brief This file specifies compile time options for how we use mfem.
 * In particular the default mfem::Ordering used for our FESpaces
 */

#pragma once

#include "mfem.hpp"

namespace serac {

/// The mfem ordering / memory layout used for finite element vectors throughout serac
constexpr auto ordering = mfem::Ordering::byVDIM;

}  // namespace serac
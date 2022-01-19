// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file loading_functions.hpp
 *
 * @brief Some simple default loading functions for the Serac driver
 */

#pragma once

#include "mfem.hpp"

namespace serac {

/**
 * @brief A function to return a zero deformation reference configuration
 *
 * @param[in] x The input spatial position
 * @param[out] y The same reference configuration for the given spatial position (y=x)
 */
void referenceConfiguration(const mfem::Vector& x, mfem::Vector& y);

/**
 * @brief A function to return a zero initial deformation
 *
 * @param[in] x The input spatial position
 * @param[out] y The zero initial deformation (y=0)
 */
void initialDeformation(const mfem::Vector& x, mfem::Vector& y);

}  // namespace serac

// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file quadrature_data.cpp
 *
 * @brief This file exists to instantiate some global QuadratureData objects
 */

#include "serac/numerics/quadrature_data.hpp"

namespace serac {

QuadratureData<Nothing> NoQData;
QuadratureData<Empty> EmptyQData;

}  // namespace serac

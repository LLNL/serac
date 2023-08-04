// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file quadrature_data.cpp
 *
 * @brief This file exists to instantiate some global QuadratureData objects
 */

#include "serac/numerics/functional/quadrature_data.hpp"

namespace serac {

/// a single instance of a QuadratureData container of `Nothing`s, since they are all interchangeable
std::shared_ptr<QuadratureData<Nothing> > NoQData;

/// a single instance of a QuadratureData container of `Empty`s, since they are all interchangeable
std::shared_ptr<QuadratureData<Empty> > EmptyQData;

}  // namespace serac

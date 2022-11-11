// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file tuple_arithmetic.hpp
 *
 * @brief Definitions of arithmetic operations on tuples of values
 */

// this file defines basic arithmetic operations on tuples of values
// so that expressions like sum1 and sum2 below are equivalent
//
// serac::tuple< foo, bar > a;
// serac::tuple< baz, qux > b;
//
// serac::tuple sum1 = a + b;
// serac::tuple sum2{serac::get<0>(a) + serac::get<0>(b), serac::get<1>(a) + serac::get<1>(b)};

#pragma once

#include <utility>

#include "tuple.hpp"
#include "tensor.hpp"

namespace serac {



}  // namespace serac

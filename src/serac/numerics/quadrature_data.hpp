// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file quadrature_data.hpp
 *
 * @brief This file contains the declaration of the structure that manages quadrature point data
 */

#pragma once

#include "mfem.hpp"

#include "axom/core.hpp"

#include "serac/serac_config.hpp"

#include "serac/infrastructure/accelerator.hpp"

namespace serac {

struct Empty{};

struct SyncableData {
  virtual void sync() = 0;
};

template < typename T >
struct QuadratureData : public SyncableData {
  void sync() final {} // ?
  SERAC_HOST_DEVICE T& operator()(const int element, const int quadrature_point) {
    return data(element, quadrature_point);
  }
  axom::Array<T, 2, axom::MemorySpace::Host> data;
};

template <>
struct QuadratureData<Empty> : public SyncableData {
  void sync() final {} // ?
  SERAC_HOST_DEVICE Empty & operator()(const int, const int) { return data; }
  Empty data;
};

extern QuadratureData<Empty> EmptyQData;

}  // namespace serac

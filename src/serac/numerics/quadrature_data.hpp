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

struct Nothing{};

struct Empty{};

struct SyncableData {
  virtual void sync() = 0;
};

template < typename T >
struct QuadratureData : public SyncableData, public axom::Array<T, 2, axom::MemorySpace::Host> {
  void sync() final {} // ?
  using axom::Array<T, 2, axom::MemorySpace::Host >::Array;
};

template <>
struct QuadratureData<Nothing> : public SyncableData {
  void sync() final {} // ?
  SERAC_HOST_DEVICE Nothing & operator()(const int, const int) { return data; }
  Nothing data;
};

template <>
struct QuadratureData<Empty> : public SyncableData {
  void sync() final {} // ?
  SERAC_HOST_DEVICE Empty & operator()(const int, const int) { return data; }
  Empty data;
};

extern QuadratureData<Nothing> NoQData;
extern QuadratureData<Empty> EmptyQData;




}  // namespace serac

namespace axom {

template <>
struct ArrayView<serac::Nothing, 2> {
  SERAC_HOST_DEVICE serac::Nothing & operator()(const int, const int) { return data; }
  serac::Nothing data;
};

template <>
struct ArrayView<serac::Empty, 2> {
  SERAC_HOST_DEVICE serac::Empty & operator()(const int, const int) { return data; }
  serac::Empty data;
};

}

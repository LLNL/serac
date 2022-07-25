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

/**
 * @brief these classes are a little confusing. These two
 * special types represent the similar (but different) cases of:
 * 
 * Nothing: for qfunctions that have no notion of quadrature data (e.g. body forces).
 *          QuadratureData<Nothing> will store no data, and `Nothing` will never appear
 *          as an argument to a q-function (it will be omitted)
 * 
 * Empty: for qfunctions associated with material models (where quadrature data is part of
 *        the interface) that do not actually need to store internal variables. QuadratureData<Empty>
 *        will also store no data, but it will still appear as an argument to the q-function
 *        (to make the material model interface consistent).
 */
struct Nothing{};

/**
 * @brief see `Nothing` for a complete description of this class and when to use it
 */
struct Empty{};

/**
 * @brief A class for storing and access user-defined types at quadrature points
 * 
 * @tparam the data type to be stored 
 */
template < typename T >
struct QuadratureData {
  QuadratureData(size_t n1, size_t n2) : stride(n2) { data = new T[n1 * n2]; }
  ~QuadratureData() { delete data; }
  SERAC_HOST_DEVICE T & operator()(size_t i, size_t j) { return data[i * stride + j]; }
  SERAC_HOST_DEVICE const T & operator()(size_t i, size_t j) const { return data[i * stride + j]; }
  T * data;
  size_t stride;
};

/**
 * @brief a specialization of the QuadratureData container, for the type `Nothing`
 * that implements the appropriate interface requirements, but does not allocate any
 * memory on the heap
 */
template <>
struct QuadratureData<Nothing> {
  SERAC_HOST_DEVICE Nothing & operator()(const size_t, const size_t) { return data; }
  SERAC_HOST_DEVICE const Nothing & operator()(const size_t, const size_t) const { return data; }
  Nothing data;
};

/**
 * @brief a specialization of the QuadratureData container, for the type `Empty`
 * that implements the appropriate interface requirements, but does not allocate any
 * memory on the heap
 */
template <>
struct QuadratureData<Empty> {
  SERAC_HOST_DEVICE Empty & operator()(const size_t, const size_t) { return data; }
  SERAC_HOST_DEVICE const Empty & operator()(const size_t, const size_t) const { return data; }
  Empty data;
};

extern std::shared_ptr< QuadratureData<Nothing> > NoQData;
extern std::shared_ptr <QuadratureData<Empty> > EmptyQData;

}  // namespace serac

// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
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
struct Nothing {
};

/**
 * @brief see `Nothing` for a complete description of this class and when to use it
 */
struct Empty {
};

template <typename T>
struct QuadratureData;

} // namespace serac

namespace axom {

template <>
class Array<serac::Nothing, 2, MemorySpace::Dynamic> {
 public:
  Array() {}
  Array(uint32_t, uint32_t) {}
};

template <>
class ArrayView<serac::Nothing, 2, MemorySpace::Dynamic> {
 public:
  ArrayView(Array<serac::Nothing, 2, MemorySpace::Dynamic> /* unused */) {}

  /// dummy accessor to satisfy interface requirements
  SERAC_HOST_DEVICE serac::Nothing& operator()(const size_t, const size_t) { return data; }

  /// dummy accessor to satisfy interface requirements
  SERAC_HOST_DEVICE const serac::Nothing& operator()(const size_t, const size_t) const { return data; }

  serac::Nothing data;
};

template <>
class Array<serac::Empty, 2, MemorySpace::Dynamic> {
 public:
  Array() {}
  Array(uint32_t, uint32_t) {}
};

template <>
class ArrayView<serac::Empty, 2, MemorySpace::Dynamic> {
 public:
  ArrayView(Array<serac::Empty, 2, MemorySpace::Dynamic> /* unused */) {}

  /// dummy accessor to satisfy interface requirements
  SERAC_HOST_DEVICE serac::Empty& operator()(const size_t, const size_t) { return data; }

  /// dummy accessor to satisfy interface requirements
  SERAC_HOST_DEVICE const serac::Empty& operator()(const size_t, const size_t) const { return data; }

  serac::Empty data;
};

} // namespace axom

namespace serac {

/**
 * @brief A class for storing and access user-defined types at quadrature points
 *
 * @tparam the data type to be stored
 *
 * @note users are not intended to create these objects directly, instead
 *       they should use the PhysicsModule::createQuadratureDataBuffer()
 */
template <typename T>
struct QuadratureData {

  using geom_array_t = std::array< uint32_t, mfem::Geometry::NUM_GEOMETRIES >;

  using tmp_t = axom::Array<T, 2>;

  /// ctor, allocates memory and sets up strides
  QuadratureData(geom_array_t elements, geom_array_t qpts_per_element, T value = T{}) { 

    constexpr std::array geometries = {
      mfem::Geometry::SEGMENT,
      mfem::Geometry::TRIANGLE,
      mfem::Geometry::SQUARE,
      mfem::Geometry::TETRAHEDRON,
      mfem::Geometry::CUBE
    };

    for (auto geom : geometries) {
      if (elements[uint32_t(geom)] > 0) {
        data[geom] = tmp_t(elements[geom], qpts_per_element[geom]);
        data[geom].fill(value);
      }
    }

  }

  axom::ArrayView<T, 2> operator[](mfem::Geometry::Type geom) {
    return axom::ArrayView<T, 2>(data.at(geom));
  }

  std::map< mfem::Geometry::Type, tmp_t > data;
};

template <>
struct QuadratureData<Nothing> {

  using geom_array_t = std::array< uint32_t, mfem::Geometry::NUM_GEOMETRIES >;

  QuadratureData() {}

  axom::ArrayView<Nothing, 2> operator[](mfem::Geometry::Type) {
    return axom::ArrayView<Nothing, 2>(data);
  }

  axom::Array<Nothing, 2, axom::MemorySpace::Dynamic> data;
};

template <>
struct QuadratureData<Empty> {

  using geom_array_t = std::array< uint32_t, mfem::Geometry::NUM_GEOMETRIES >;

  QuadratureData() {}

  axom::ArrayView<Empty, 2> operator[](mfem::Geometry::Type) {
    return axom::ArrayView<Empty, 2>(data);
  }

  axom::Array<Empty, 2, axom::MemorySpace::Dynamic> data;
};

extern std::shared_ptr<QuadratureData<Nothing> > NoQData;
extern std::shared_ptr<QuadratureData<Empty> >   EmptyQData;

}  // namespace serac

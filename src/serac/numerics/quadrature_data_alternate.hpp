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

#include "serac/infrastructure/variant.hpp"

namespace serac {

template < typename T >
class QuadratureData {

}

/**
 * @brief Policy class for implementing indexing of quadrature point data
 */
template <typename T, typename Derived>
class QuadratureDataImpl {
public:
  /**
   * @brief Retrieves the data for a given quadrature point
   * @param[in] element_idx The index of the desired element within the mesh
   * @param[in] q_idx The index of the desired quadrature point within the element
   */
  SERAC_HOST_DEVICE T& operator()(const int element_idx, const int q_idx);

private:
  SERAC_HOST_DEVICE Derived& asDerived() { return static_cast<Derived&>(*this); }
  SERAC_HOST_DEVICE const Derived& asDerived() const { return static_cast<const Derived&>(*this); }
};

/**
 * @brief Stores instances of user-defined type for each quadrature point in a mesh
 * @tparam T The type of the per-qpt data
 * @pre T must be default-constructible
 * @pre T must be trivially copyable (due to the use of memcpy for type punning)
 */
template <typename T>
class QuadratureData : public SyncableData, public QuadratureDataImpl<T, QuadratureData<T>> {
public:
  /**
   * @brief Constructs using a mesh and polynomial order
   * @param[in] mesh The mesh for which quadrature-point data should be stored
   * @param[in] p The polynomial order of the associated finite elements
   * @param[in] alloc Flag to allocate the underlying data
   */
  QuadratureData(mfem::Mesh& mesh, const int p, const bool alloc = true);

// Turn off null dereference warnings for GCC
// TODO Fix the underlying possible nullptr dereference warning with the `MaybeOwnedPointer` type.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"

  /**
   * @brief Constructs from an existing quadrature function
   * @param[in] qfunc The QuadratureFunction with existing quadrature
   * point data
   *
   * @pre @a qfunc must be created via an instance of this class, i.e.,
   * this constructor is intended to be used as part of a save/restart
   */
  QuadratureData(mfem::QuadratureFunction& qfunc)
      : qspace_(qfunc.GetSpace()),
        qfunc_(&qfunc),
        data_(static_cast<std::size_t>(detail::retrieve(qfunc_).Size() / stride_)),
        offsets_(static_cast<std::size_t>(qfunc.GetSpace()->GetNE() + 1))
  {
    std::memcpy(offsets_.data(), detail::quadSpaceOffsets(detail::retrieve(qspace_)),
                static_cast<std::size_t>(qfunc.GetSpace()->GetNE() + 1) * sizeof(int));
    const double* qfunc_ptr = detail::retrieve(qfunc_).GetData();
    int           j         = 0;
    T*            data_ptr  = data_.data();
    for (int i = 0; i < detail::retrieve(qfunc_).Size(); i += stride_) {
      // The only legal (portable, defined) way to do type punning in C++
      std::memcpy(data_ptr + j, qfunc_ptr + i, sizeof(T));
      j++;
    }
  }

#pragma GCC diagnostic pop

  // When a QuadratureData instance is managed by StateManager, we don't
  // ever want to copy from or move from that instance.  This is sort of
  // overkill when QuadratureData objects are created outside of the StateManager,
  // but this case should be fairly rare.
  QuadratureData(const QuadratureData&) = delete;
  QuadratureData(QuadratureData&&)      = delete;

  /**
   * @brief Assigns an item to each quadrature point
   * @param[in] item The item to assign
   */
  QuadratureData& operator=(const T& item);

  /**
   * @brief Iterator to the data for the first quadrature point
   */
  auto begin() { return data_.data(); }
  /// @overload
  auto begin() const { return data_.data(); }
  /**
   * @brief Iterator to one element past the data for the last quadrature point
   */
  auto end() { return data_.data() + data_.size(); }
  /// @overload
  auto end() const { return data_.data() + data_.size(); }

  /**
   * @brief Get the underlying MFEM quadrature function
   *
   * @return The underlying quadrature function MFEM-based data container
   */
  mfem::QuadratureFunction& QFunc() { return detail::retrieve(qfunc_); }

  /**
   * @brief Synchronizes data from the stored vector<T> to the raw double*
   * array used by the underlying mfem::QuadratureFunction
   *
   * Used for saving to a file - MFEMSidreDataCollection
   * (and by extension mfem::DataCollection's interface) only allow for
   * quadrature-point-specific data via mfem::QuadratureFunction, so this logic
   * is needed to glue together a generic array of data with that class
   */
  void sync() override
  {
    double*  qfunc_ptr = detail::retrieve(qfunc_).GetData();
    int      j         = 0;
    const T* data_ptr  = data_.data();
    for (int i = 0; i < detail::retrieve(qfunc_).Size(); i += stride_) {
      // The only legal (portable, defined) way to do type punning in C++
      std::memcpy(qfunc_ptr + i, data_ptr + j, sizeof(T));
      j++;
    }
  }

  /// @brief Returns the underlying data array
  SERAC_HOST_DEVICE T* data() { return data_.data(); }
  /// @brief Returns the element offsets array
  SERAC_HOST_DEVICE const int* offsets() const { return offsets_.data(); }

private:
  /**
   * @brief Storage layout of @p qfunc_ containing mesh and polynomial order info
   */
  detail::MaybeOwningPointer<mfem::QuadratureSpace> qspace_;
  /**
   * @brief Per-quadrature point data, stored as array of doubles for compatibility with Sidre
   * @note It may be possible to reduce memory pressure by only constructing the qfunc immediately prior to saving
   */
  detail::MaybeOwningPointer<mfem::QuadratureFunction> qfunc_;
  /**
   * @brief The actual data
   */
  UnifiedArray<T> data_;
  /**
   * @brief A copy of the element_offsets member from mfem::QuadratureSpace
   */
  UnifiedArray<int> offsets_;
  /**
   * @brief The stride of the array
   */
  static constexpr int stride_ = detail::ceil(sizeof(T) / static_cast<double>(sizeof(double)));
};

struct Empty{};

template <>
class QuadratureData<Empty> {
public:
  SERAC_HOST_DEVICE Empty & operator()(const int, const int) { return data; }
  Empty data;
};

/**
 * @brief "Dummy" specialization, intended to be used as a sentinel
 * This is used as the default argument when a reference to a @p QuadratureData is used as a function
 * argument. By comparing the argument to the dummy instance of this class, functions to easily check
 * if the user has passed in a "real" @p QuadratureData.
 */
template <>
class QuadratureData<void> {
public:
  /// @brief Dummy data access
  SERAC_HOST_DEVICE std::nullptr_t operator()(const int, const int) { return nullptr; }
};

// A dummy global so that lvalue references can be bound to something of type QData<void>
extern QuadratureData<void> dummy_qdata;

/**
 * @brief Stores instances of user-defined type for each quadrature point in a mesh
 * @tparam T The type of the per-qpt data
 * @pre T must be default-constructible
 * @pre T must be trivially copyable (due to the use of memcpy for type punning)
 */
template <typename T>
class QuadratureDataView : public QuadratureDataImpl<T, QuadratureDataView<T>> {
public:
  /**
   * @brief Constructs a QuadratureDataView from a QuadratureData
   * @param[in] quad_data The QuadratureData to take a view of
   */
  QuadratureDataView(QuadratureData<T>& quad_data) : data_(quad_data.data()), offsets_(quad_data.offsets()) {}
  /// @brief Returns the underlying data array
  SERAC_HOST_DEVICE T* data() { return data_; }
  /// @brief Returns the element offsets array
  SERAC_HOST_DEVICE const int* offsets() const { return offsets_; }

private:
  T*         data_    = nullptr;
  const int* offsets_ = nullptr;
};

/**
 * @brief "Dummy" specialization, intended to be used as a sentinel
 * This is used as the default argument when a reference to a @p QuadratureData is used as a function
 * argument. By comparing the argument to the dummy instance of this class, functions to easily check
 * if the user has passed in a "real" @p QuadratureData.
 */
template <>
class QuadratureDataView<void> {
public:
  /// @brief Constructs a dummy view
  QuadratureDataView(QuadratureData<void>& = dummy_qdata) {}
  /// @brief Dummy data access
  SERAC_HOST_DEVICE std::nullptr_t operator()(const int, const int) { return nullptr; }
};

// A dummy global so that lvalue references can be bound to something of type QData<void>
extern QuadratureDataView<void> dummy_qdata_view;

// Hijacks the "vdim" parameter (number of doubles per qpt) to allocate the correct amount of storage
template <typename T>
QuadratureData<T>::QuadratureData(mfem::Mesh& mesh, const int p, const bool alloc)
    : qspace_(std::make_unique<mfem::QuadratureSpace>(&mesh, p + 1)),
      // When left unallocated, the allocation can happen inside the datastore
      // Use a raw pointer here when unallocated, lifetime will be managed by the DataCollection
      qfunc_(detail::initialQuadFunc(&detail::retrieve(qspace_), alloc, stride_)),
      data_(static_cast<std::size_t>(detail::retrieve(qfunc_).Size() / stride_)),
      offsets_(static_cast<std::size_t>(mesh.GetNE() + 1))
{
  // To avoid violating C++'s strict aliasing rule we need to std::memcpy a default-constructed object
  // See e.g. https://gist.github.com/shafik/848ae25ee209f698763cffee272a58f8
  // also https://en.cppreference.com/w/cpp/numeric/bit_cast
  // also https://chromium.googlesource.com/chromium/src/base/+/refs/heads/master/bit_cast.h
  static_assert(std::is_default_constructible_v<T>, "Must be able to default-construct the stored type");
  static_assert(std::is_trivially_copyable_v<T>, "Uses memcpy - requires trivial copies");
  // We cannot avoid storing a copy of the offsets array in the general case,
  // but if we know the number of qpts per element at compile time we don't need to store offsets
  std::memcpy(offsets_.data(), detail::quadSpaceOffsets(detail::retrieve(qspace_)),
              static_cast<std::size_t>(mesh.GetNE() + 1) * sizeof(int));
}

template <typename T>
QuadratureData<T>& QuadratureData<T>::operator=(const T& item)
{
  for (auto& datum : data_) {
    datum = item;
  }
  return *this;
}

template <typename T, typename Derived>
SERAC_HOST_DEVICE T& QuadratureDataImpl<T, Derived>::operator()(const int element_idx, const int q_idx)
{
  const auto idx = asDerived().offsets()[static_cast<std::size_t>(element_idx)] + q_idx;
  return asDerived().data()[static_cast<std::size_t>(idx)];
}

}  // namespace serac

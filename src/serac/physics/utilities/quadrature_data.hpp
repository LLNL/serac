// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
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

#include "serac/physics/utilities/variant.hpp"

namespace serac {

namespace detail {

/**
 * @brief A simple constexpr implementation of @p ceil
 * @param[in] num The number to "round up"
 * @pre num must be in the range [0, std::numeric_limits<int>::max())
 */
constexpr int ceil(const double num)
{
  const auto as_int = static_cast<int>(num);
  return (static_cast<double>(as_int) == num) ? as_int : as_int + 1;
}

/**
 * @brief Helper function for creating an mfem::QuadratureFunction in both restart and not-restart scenarios
 * @param[in] space The QSpace to construct the mfem::QuadratureFunction with
 * @param[in] alloc_qf Whether to allocate the mfem::QuadratureFunction - if this is a non-restart run, we delay the
 * allocation so it can be taken care of inside MFEMSidreDataCollection
 * @param[in] stride The stride of the array (number of doubles to allocate per point)
 */
inline MaybeOwningPointer<mfem::QuadratureFunction> initialQuadFunc(mfem::QuadratureSpace* space, const bool alloc_qf,
                                                                    const int stride)
{
  if (alloc_qf) {
    return std::make_unique<mfem::QuadratureFunction>(space, stride);
  } else {
    return new mfem::QuadratureFunction(space, nullptr, stride);
  }
}

}  // namespace detail

/**
 * @brief A shim class for describing the interface of something that can be synced
 *
 * Because @p QuadratureData is templated, we cannot store a collection of them of arbitrary type.
 * To support this for the purposes of iterating over them + synchronizing them within @p StateManager::Save
 * we need to introduce a non-templated layer of indirection, i.e., this class.  We store a collection of these
 * references to @p QuadratureData<T> (of varying @p T ) and call their virtual @p sync() methods to perform
 * the type-punning synchronization such that the underlying @p mfem::QuadratureFunction contains the updated data
 * that gets saved to disk by @p MFEMSidreDataCollection.
 *
 * Further info:
 * In practice this is used by @p StateManager so that the @p T[] stored by @p QuadratureData
 * can be synced (copied) to the @p double[] owned by its underlying @p mfem::QuadratureFunction
 * immediately prior to saving the data to disk.  That is, the interface for this class
 * is intended to capture the @p T[] -> @p double[] action required to use @p QuadratureData
 * with @p mfem::DataCollection's interface for quadrature point data (i.e., through @p mfem::QuadratureFunction )
 */
class SyncableData {
public:
  virtual ~SyncableData() = default;
  virtual void sync()     = 0;
};

/**
 * @brief Stores instances of user-defined type for each quadrature point in a mesh
 * @tparam T The type of the per-qpt data
 * @pre T must be default-constructible (TODO: Do we want to allow non-default constructible types?)
 * @pre T must be trivially copyable (due to the use of memcpy for type punning)
 */
template <typename T>
class QuadratureData : public SyncableData {
public:
  /**
   * @brief Constructs using a mesh and polynomial order
   * @param[in] mesh The mesh for which quadrature-point data should be stored
   * @param[in] p The polynomial order of the associated finite elements
   */
  QuadratureData(mfem::Mesh& mesh, const int p, const bool alloc = true);

  /**
   * @brief Constructs from an existing quadrature function
   * @param[in] qfunc The QuadratureFunction with existing quadrature
   * point data
   *
   * @pre @a qfunc must be created via an instance of this class, i.e.,
   * this constructor is intended to be used as part of a save/restart
   */
  QuadratureData(mfem::QuadratureFunction& qfunc)
      : qspace_(qfunc.GetSpace()), qfunc_(&qfunc), data_(static_cast<std::size_t>(qfunc.Size() / stride_))
  {
    const double* qfunc_ptr = detail::retrieve(qfunc_).GetData();
    int           j         = 0;
    T*            data_ptr  = data_.data();
    for (int i = 0; i < detail::retrieve(qfunc_).Size(); i += stride_) {
      // The only legal (portable, defined) way to do type punning in C++
      std::memcpy(data_ptr + j, qfunc_ptr + i, sizeof(T));
      j++;
    }
  }

  // When a QuadratureData instance is managed by StateManager, we don't
  // ever want to copy from or move from that instance.  This is sort of
  // overkill when QuadratureData objects are created outside of the StateManager,
  // but this case should be fairly rare.
  QuadratureData(const QuadratureData&) = delete;
  QuadratureData(QuadratureData&&)      = delete;

  /**
   * @brief Retrieves the data for a given quadrature point
   * @param[in] element_idx The index of the desired element within the mesh
   * @param[in] q_idx The index of the desired quadrature point within the element
   */
  T& operator()(const int element_idx, const int q_idx);

  /**
   * @brief Assigns an item to each quadrature point
   * @param[in] item The item to assign
   */
  QuadratureData& operator=(const T& item);

  /**
   * @brief Iterator to the data for the first quadrature point
   */
  auto begin() { return data_.begin(); }
  /// @overload
  auto begin() const { return data_.begin(); }
  /**
   * @brief Iterator to one element past the data for the last quadrature point
   */
  auto end() { return data_.end(); }
  /// @overload
  auto end() const { return data_.end(); }

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

private:
  /**
   * @brief Storage layout of @p qfunc_ containing mesh and polynomial order info
   */
  detail::MaybeOwningPointer<mfem::QuadratureSpace> qspace_;
  /**
   * @brief Per-quadrature point data, stored as array of doubles for compatibility with Sidre
   */
  detail::MaybeOwningPointer<mfem::QuadratureFunction> qfunc_;
  /**
   * @brief The actual data
   */
  std::vector<T> data_;
  /**
   * @brief The stride of the array
   */
  static constexpr int stride_ = detail::ceil(sizeof(T) / static_cast<double>(sizeof(double)));
};

/**
 * @brief "Dummy" specialization, intended to be used as a sentinel
 * This is used as the default argument when a reference to a @p QuadratureData is used as a function
 * argument. By comparing the argument to the dummy instance of this class, functions to easily check
 * if the user has passed in a "real" @p QuadratureData.
 */
template <>
class QuadratureData<void> {
};

// A dummy global so that lvalue references can be bound to something of type QData<void>
// FIXME: There's probably a cleaner way to do this, it's technically a non-const global
// but it's not really mutable because no operations are defined for it
extern QuadratureData<void> dummy_qdata;

// Hijacks the "vdim" parameter (number of doubles per qpt) to allocate the correct amount of storage
template <typename T>
QuadratureData<T>::QuadratureData(mfem::Mesh& mesh, const int p, const bool alloc)
    : qspace_(std::make_unique<mfem::QuadratureSpace>(&mesh, p + 1)),
      // When left unallocated, the allocation can happen inside the datastore
      // Use a raw pointer here when unallocated, lifetime will be managed by the DataCollection
      qfunc_(detail::initialQuadFunc(&detail::retrieve(qspace_), alloc, stride_)),
      data_(static_cast<std::size_t>(detail::retrieve(qfunc_).Size() / stride_))
{
  // To avoid violating C++'s strict aliasing rule we need to std::memcpy a default-constructed object
  // See e.g. https://gist.github.com/shafik/848ae25ee209f698763cffee272a58f8
  // also https://en.cppreference.com/w/cpp/numeric/bit_cast
  // also https://chromium.googlesource.com/chromium/src/base/+/refs/heads/master/bit_cast.h
  static_assert(std::is_default_constructible_v<T>, "Must be able to default-construct the stored type");
  static_assert(std::is_trivially_copyable_v<T>, "Uses memcpy - requires trivial copies");
}

template <typename T>
T& QuadratureData<T>::operator()(const int element_idx, const int q_idx)
{
  // A view into the quadrature point data
  mfem::Vector view;
  // Use the existing MFEM offset calculation logic instead of reimplementing it here
  // Avoids making this code dependent on mfem::QuadratureSpace impl
  detail::retrieve(qfunc_).GetElementValues(element_idx, q_idx, view);
  double*    end_ptr   = view.GetData();
  double*    start_ptr = detail::retrieve(qfunc_).GetData();
  const auto idx       = static_cast<std::size_t>(end_ptr - start_ptr) / stride_;
  return data_[static_cast<std::size_t>(idx)];
}

template <typename T>
QuadratureData<T>& QuadratureData<T>::operator=(const T& item)
{
  data_.assign(data_.size(), item);
  return *this;
}

}  // namespace serac

// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file quadrature_data.hpp
 *
 * @brief The definition of the QuadratureData class
 */

#pragma once

#include "mfem.hpp"

// namespace serac {

/**
 * @brief Type-punning iterator using same method as std::bit_cast
 * @tparam T The type to pun
 * @note This class should be used carefully as changes to the object are
 * not propagated back to the underlying pointer until the destructor is called
 */
template <typename T>
class PunIterator {
  // Semantics get too confusing
  static_assert(!std::is_pointer_v<T>, "Raw pointer types not supported");

public:
  /**
   * @brief Constructs an "empty" iterator
   */
  PunIterator() = default;
  /**
   * @brief Constructs an iterator
   * @param[in] ptr A pointer to an element
   * @param[in] end_ptr A pointer to one-past-the-end of the container
   */
  PunIterator(void* ptr, void* end_ptr) : ptr_(ptr), end_ptr_(end_ptr) {}

  /**
   * @brief Cleans up by storing @p obj_ back into @p ptr_ to maintain coherency
   */
  ~PunIterator()
  {
    if (ptr_ && (ptr_ != end_ptr_)) {
      std::memcpy(ptr_, &obj_, sizeof(T));
    }
  }

  /**
   * @brief Returns the stored object
   * @note Changes to the object will not be propagated until the destructor is called
   */
  T& operator*()
  {
    std::memcpy(&obj_, ptr_, sizeof(T));
    return obj_;
  }

  /**
   * @brief Advances the iterator
   */
  PunIterator& operator++()
  {
    // This is permissible because we're not actually dereferencing ptr_
    // as a T*, just using it for arithmetic
    ptr_ = static_cast<T*>(ptr_) + 1;
    return *this;
  }

  /**
   * @brief Compares two iterators for equality
   */
  bool operator!=(const PunIterator& other) { return ptr_ != other.ptr_; }

private:
  /**
   * @brief Pointer to the current element
   */
  void* ptr_ = nullptr;
  /**
   * @brief Pointer to one-past-the-end of the container
   */
  void* end_ptr_ = nullptr;
  /**
   * @brief A mirror of the data in ptr_
   */
  T obj_;
};

/**
 * @brief Stores instances of user-defined type for each quadrature point in a mesh
 * @tparam T The type of the per-qpt data
 * @pre T must be default-constructible (TODO: Do we want to allow non-default constructible types?)
 * @pre T must be trivially copyable (due to the use of memcpy for type punning)
 */
template <typename T>
class QuadratureData {
public:
  /**
   * @brief Constructs using a mesh and polynomial order
   * @param[in] mesh The mesh for which quadrature-point data should be stored
   * @param[in] p The polynomial order of the associated finite elements
   */
  QuadratureData(mfem::Mesh& mesh, const int p);

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
  PunIterator<T> begin()
  {
    // WARNING: THIS IS REQUIRED BEFORE ANY ACCESSES AND MUST BE PROPAGATED TO OTHER ACCESSORS
    proxy_.reset();
    return {qfunc_.GetData(), qfunc_.GetData() + qfunc_.Size()};
  }
  /**
   * @brief Iterator to one element past the data for the last quadrature point
   */
  PunIterator<T> end() { return {qfunc_.GetData() + qfunc_.Size(), qfunc_.GetData() + qfunc_.Size()}; }

private:
  // FIXME: These will probably need to be MaybeOwningPointers
  // See https://github.com/LLNL/axom/pull/433
  /**
   * @brief Storage layout of @p qfunc_ containing mesh and polynomial order info
   */
  mfem::QuadratureSpace qspace_;
  /**
   * @brief Per-quadrature point data, stored as array of doubles for compatibility with Sidre
   */
  mfem::QuadratureFunction qfunc_;
  /**
   * @brief Provides reference-like semantics through a standard-compliant type pun
   */
  std::optional<PunIterator<T>> proxy_;
  /**
   * @brief The stride of the array
   */
  constexpr static int stride_ = sizeof(T) / sizeof(double);
};

/**
 * @brief "Dummy" specialization, intended to be used as sentinel
 */
template <>
class QuadratureData<void> {
};

// A dummy global so that lvalue references can be bound to something of type QData<void>
// FIXME: There's probably a cleaner way to do this, it's technically a non-const global
// but it's not really mutable because no operations are defined for it
QuadratureData<void> dummy_qdata;

// Hijacks the "vdim" parameter (number of doubles per qpt) to allocate the correct amount of storage
template <typename T>
QuadratureData<T>::QuadratureData(mfem::Mesh& mesh, const int p) : qspace_(&mesh, p + 1), qfunc_(&qspace_, stride_)
{
  // To avoid violating C++'s strict aliasing rule we need to std::memcpy a default-constructed object
  // See e.g. https://gist.github.com/shafik/848ae25ee209f698763cffee272a58f8
  // also https://en.cppreference.com/w/cpp/numeric/bit_cast
  // also https://chromium.googlesource.com/chromium/src/base/+/refs/heads/master/bit_cast.h
  static_assert(std::is_default_constructible_v<T>, "Must be able to default-construct the stored type");
  static_assert(std::is_trivially_copyable_v<T>, "Uses memcpy - requires trivial copies");
  T       default_constructed;  // Will be memcpy'd into each element
  double* ptr = qfunc_.GetData();
  for (int i = 0; i < qfunc_.Size(); i += stride_) {
    // The only legal (portable, defined) way to do type punning in C++
    // Would be illegal to just placement-new construct a T here
    std::memcpy(ptr + i, &default_constructed, sizeof(T));
  }
}

template <typename T>
T& QuadratureData<T>::operator()(const int element_idx, const int q_idx)
{
  // A view into the quadrature point data
  mfem::Vector view;
  qfunc_.GetElementValues(element_idx, q_idx, view);
  proxy_.emplace(view.GetData(), qfunc_.GetData() + qfunc_.Size());
  return *proxy_.value();
}

template <typename T>
QuadratureData<T>& QuadratureData<T>::operator=(const T& item)
{
  double* ptr = qfunc_.GetData();
  for (int i = 0; i < qfunc_.Size(); i += stride_) {
    std::memcpy(ptr + i, &item, sizeof(T));
  }
  return *this;
}
// }  // namespace serac

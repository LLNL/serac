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
  auto begin() { return data_.begin(); }
  /// @overload
  auto begin() const { return data_.begin(); }
  /**
   * @brief Iterator to one element past the data for the last quadrature point
   */
  auto end() { return data_.end(); }
  /// @overload
  auto end() const { return data_.end(); }

  void syncFromQFunc()
  {
    const double* qfunc_ptr = qfunc_.GetData();
    int           j         = 0;
    T*            data_ptr  = data_.data();
    for (int i = 0; i < qfunc_.Size(); i += stride_) {
      // The only legal (portable, defined) way to do type punning in C++
      std::memcpy(data_ptr + j, qfunc_ptr + i, sizeof(T));
      j++;
    }
  }

  void syncToQFunc()
  {
    double*  qfunc_ptr = qfunc_.GetData();
    int      j         = 0;
    const T* data_ptr  = data_.data();
    for (int i = 0; i < qfunc_.Size(); i += stride_) {
      // The only legal (portable, defined) way to do type punning in C++
      std::memcpy(qfunc_ptr + i, data_ptr + j, sizeof(T));
      j++;
    }
  }

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

  std::vector<T> data_;
  /**
   * @brief The stride of the array
   */
  static constexpr int stride_ = sizeof(T) / sizeof(double);
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
QuadratureData<T>::QuadratureData(mfem::Mesh& mesh, const int p)
    : qspace_(&mesh, p + 1), qfunc_(&qspace_, stride_), data_(qfunc_.Size() / stride_)
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
  qfunc_.GetElementValues(element_idx, q_idx, view);
  double*    end_ptr   = view.GetData();
  double*    start_ptr = qfunc_.GetData();
  const auto idx       = (end_ptr - start_ptr) / stride_;
  return data_[idx];
}

template <typename T>
QuadratureData<T>& QuadratureData<T>::operator=(const T& item)
{
  data_.assign(data_.size(), item);
  return *this;
}

// }  // namespace serac

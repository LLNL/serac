// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file array_4D.hpp
 *
 * @brief MFEM extension for 4D arrays
 *
 */

#pragma once

#include "serac/infrastructure/logger.hpp"

#include "mfem.hpp"

/**
 * The Serac mfem extension namespace
 */
namespace serac::mfem_ext {

/**
 * @brief A 4D array class for tensor operators
 * @tparam T The base container type, e.g. a double
 * @note This class will get replaced by a more rigorous tensor class soon
 */
template <class T>
class Array4D {
public:
  /**
   * @brief Construct an empty 4D array object
   *
   */
  Array4D() { N2_ = N3_ = N4_ = 0; }

  /**
   * @brief Construct a sized 4D array object
   *
   * @param[in] n1 dimension 1
   * @param[in] n2 dimension 2
   * @param[in] n3 dimension 3
   * @param[in] n4 dimension 4
   */
  Array4D(int n1, int n2, int n3, int n4) : array1d_(n1 * n2 * n3 * n4)
  {
    N2_ = n2;
    N3_ = n3;
    N4_ = n4;
  }

  /**
   * @brief Resize the 4D array
   *
   * @param[in] n1 dimension 1
   * @param[in] n2 dimension 2
   * @param[in] n3 dimension 3
   * @param[in] n4 dimension 4
   */
  void SetSize(int n1, int n2, int n3, int n4)
  {
    array1d_.SetSize(n1 * n2 * n3 * n4);
    N2_ = n2;
    N3_ = n3;
    N4_ = n4;
  }

  /**
   * @brief Const accessor to an entry in the 4D array
   *
   * @param[in] i first index
   * @param[in] j second index
   * @param[in] k third index
   * @param[in] l fourth index
   * @return a const reference to the value at array(i,j,k,l)
   */
  inline const T& operator()(int i, int j, int k, int l) const;

  /**
   * @brief Accessor to an entry in the 4D array
   *
   * @param[in] i first index
   * @param[in] j second index
   * @param[in] k third index
   * @param[in] l fourth index
   * @return a reference to the value at array(i,j,k,l)
   */
  inline T& operator()(int i, int j, int k, int l);

  /**
   * @brief Set the full 4D array to a constant value
   *
   * @param[in] a the value to set as every element in the array
   */
  inline void operator=(const T& a) { array1d_ = a; }

private:
  /**
   * @brief The underlying 1D array for contiguous memory layout
   *
   */
  mfem::Array<T> array1d_;

  /**
   * @brief The sizes of index 2, 3, and 4
   *
   */
  int N2_, N3_, N4_;
};

template <class T>
inline const T& Array4D<T>::operator()(int i, int j, int k, int l) const
{
  SLIC_ASSERT_MSG(
      i >= 0 && i < array1d_.Size() / N2_ / N3_ / N4_ && j >= 0 && j < N2_ && k >= 0 && k < N3_ && k >= 0 && k < N4_,
      "Array4D: invalid access of element.");
  return array1d_[((i * N2_ + j) * N3_ + k) * N4_ + l];
}

template <class T>
inline T& Array4D<T>::operator()(int i, int j, int k, int l)
{
  SLIC_ASSERT_MSG(
      i >= 0 && i < array1d_.Size() / N2_ / N3_ / N4_ && j >= 0 && j < N2_ && k >= 0 && k < N3_ && k >= 0 && k < N4_,
      "Array4D: invalid access of element.");
  return array1d_[((i * N2_ + j) * N3_ + k) * N4_ + l];
}

}  // namespace serac::mfem_ext

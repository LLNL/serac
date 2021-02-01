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

#include "mfem.hpp"

/**
 * The Serac mfem extension namespace
 */
namespace serac::mfem_ext {

template <class T>
class Array4D
{
private:
   mfem::Array<T> array1d_;
   int N2_, N3_, N4_;

public:
   Array4D() { N2_ = N3_ = N4_ = 0; }
   Array4D(int n1, int n2, int n3, int n4)
      : array1d_(n1*n2*n3*n4) { N2_ = n2; N3_ = n3; N4_ = n4; }

   void SetSize(int n1, int n2, int n3, int n4)
   { array1d_.SetSize(n1*n2*n3*n4); N2_ = n2; N3_ = n3; N4_ = n4; }

   inline const T &operator()(int i, int j, int k, int l) const;
   inline       T &operator()(int i, int j, int k, int l);
};

template <class T>
inline const T &Array4D<T>::operator()(int i, int j, int k, int l) const
{
   SLIC_ASSERT_MSG(i >= 0 && i < array1d_.Size() / N2_ / N3_ / N4_ && j >= 0 && j < N2_
               && k >= 0 && k < N3_ && k >= 0 && k < N4_,
               "Array4D: invalid access of element.");
   return array1d_[((i*N2_+j)*N3_+k)*N4_+l];
}

template <class T>
inline T &Array4D<T>::operator()(int i, int j, int k, int l)
{
   SLIC_ASSERT_MSG(i >= 0 && i < array1d_.Size() / N2_ / N3_ / N4_ && j >= 0 && j < N2_
               && k >= 0 && k < N3_ && k >= 0 && k < N4_,
               "Array4D: invalid access of element.");
   return array1d_[((i*N2_+j)*N3_+k)*N4_+l];  
}

} // namespace mfem_ext

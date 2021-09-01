// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file arrays.hpp
 *
 * @brief implementations of 2D and 3D arrays and views to be used
 * temporarily while axom::Array is being developed
 */
#include <iostream>
#include <vector>

// clang-format off
namespace serac {

  template < int dimension >
  struct Indexable;

  template <>
  struct Indexable<2>{
    Indexable() = default;
    Indexable(int n1, int n2) : strides{n2}, shape{n1, n2} {}
    auto index(int i, int j) { return i * strides + j; }
    int strides;
    int shape[2];
  };

  template <>
  struct Indexable<3>{
    Indexable() = default;
    Indexable(int n1, int n2, int n3) : strides{n2 * n3, n3}, shape{n1, n2, n3} {}
    auto index(int i, int j, int k) { return i * strides[0] + j * strides[1] + k]; }
    int strides[2];
    int shape[3];
  };


  template < typename T >
  struct Indexable< T, 3 >{
    Indexable(int n1, int n2, int n3) : strides{n2 * n3, n3} {}
    auto & operator()(int i, int j, int k) { return values[i * strides[0] + j * strides[1] + k]; }
    const auto & operator()(int i, int j, int k) const { return values[i * strides[0] + j * strides[1] + k]; }
    int strides[2];
    T values;
  };

  template < typename T >
  struct Array2D : Indexable< std::vector<T>, 2 >{
    Array2D(int n1, int n2) : Indexable< std::vector<T>, 2 >(n1, n2) { 
      Indexable< std::vector<T>, 2 >::values.resize(n1, n2); 
    }
  };

  template < typename T >
  struct Array3D : public Indexable< std::vector<T>, 3 >{
    Array3D(int n1, int n2, int n3) : Indexable< std::vector<T>, 3 >(n1, n2, n3) { 
      Indexable< std::vector<T>, 3 >::values.resize(n1, n2, n3); 
    }
  };

  template < typename T >
  struct View2D : Indexable< T *, 2 >{ using Indexable< T *, 2 >::Indexable; };

  template < typename T >
  struct View3D : Indexable< T *, 3 >{ using Indexable< T *, 3 >::Indexable; };

}
// clang-format on

// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file array.hpp
 *
 * @brief a placeholder that implements some basic multdimensional array stuff until axom::Array is ready
 */

//! @cond suppress doxygen warnings for this file, as it is temporary

#pragma once

#include <stddef.h>

#include <iostream>
#include <algorithm>

#include "serac/infrastructure/accelerator.hpp"

namespace serac {

namespace detail {

/**
 * @tparam T the data type to allocate memory for
 * @tparam exec which memory space to use (CPU or GPU)
 * @param n how many items to allocate memory for
 *
 * @brief an abstraction for memory allocation for different types,
 *   and on different memory spaces
 */
template <typename T, ExecutionSpace exec>
T* allocate(size_t n)
{
  if constexpr (exec == ExecutionSpace::CPU) {
    return static_cast<T*>(malloc(n * sizeof(T)));
  }

#if defined(__CUDACC__)
  if constexpr (exec == ExecutionSpace::GPU) {
    T* ptr;
    cudaMalloc(&ptr, sizeof(T) * n);
    return ptr;
  }

  if constexpr (exec == ExecutionSpace::Dynamic) {
    T* ptr;
    cudaMallocManaged(&ptr, sizeof(T) * n);
    return ptr;
  }
#endif
}

/**
 * @tparam exec which memory space the memory belongs to (CPU or GPU)
 * @tparam T the data type to deallocate memory for
 *
 * @brief an abstraction for memory deallocation for different types,
 *   and on different memory spaces
 */
template <ExecutionSpace exec, typename T>
void deallocate(T* ptr)
{
  if constexpr (exec == ExecutionSpace::CPU) {
    free(ptr);
  }

#if defined(__CUDACC__)
  if constexpr (exec == ExecutionSpace::GPU || exec == ExecutionSpace::Dynamic) {
    cudaFree(ptr);
  }
#endif
}

/**
 * @tparam src_exec the memory space where src_begin and src_end ptrs reside
 * @tparam dst_exec the memory space where dst_begin resides
 *
 * @brief abstraction for copying memory between pointers. Calls either std::copy
 * or cudaMemcpy as appropriate for the execution spaces of src, dst
 */
template <ExecutionSpace src_exec, ExecutionSpace dst_exec, typename T>
void copy(T* src_begin, T* src_end, T* dst_begin)
{
  if constexpr (src_exec == ExecutionSpace::CPU && dst_exec == ExecutionSpace::CPU) {
    std::copy(src_begin, src_end, dst_begin);
  }

#if defined(__CUDACC__)

  if constexpr (src_exec == ExecutionSpace::CPU && dst_exec == ExecutionSpace::GPU) {
    size_t num_bytes = (src_end - src_begin) * sizeof(T);
    cudaMemcpy(dst_begin, src_begin, num_bytes, cudaMemcpyHostToDevice);
  }

  if constexpr (src_exec == ExecutionSpace::GPU && dst_exec == ExecutionSpace::CPU) {
    size_t num_bytes = (src_end - src_begin) * sizeof(T);
    cudaMemcpy(dst_begin, src_begin, num_bytes, cudaMemcpyDeviceToHost);
  }

  if constexpr (src_exec == ExecutionSpace::GPU && dst_exec == ExecutionSpace::GPU) {
    size_t num_bytes = (src_end - src_begin) * sizeof(T);
    cudaMemcpy(dst_begin, src_begin, num_bytes, cudaMemcpyDeviceToDevice);
  }
#endif
}

/// @brief a class responsible for generating offsets associated with a d-dimensional multi-index
template <size_t d>
struct Indexable;

template <>
struct Indexable<1> {
  size_t sizes;
  Indexable(size_t n1) : sizes{n1} {}
  SERAC_HOST_DEVICE size_t index(size_t i) const { return i; }
  size_t                   size(int = 0) const { return sizes; }
};

template <>
struct Indexable<2> {
  size_t sizes[2];
  size_t stride;
  Indexable(size_t n1, size_t n2) : sizes{n1, n2}, stride{n2} {}
  SERAC_HOST_DEVICE size_t index(size_t i, size_t j) const { return i * stride + j; }
  size_t                   size(int i = 0) { return sizes[i]; }
};

template <>
struct Indexable<3> {
  size_t sizes[3];
  size_t strides[2];
  Indexable(size_t n1, size_t n2, size_t n3) : sizes{n1, n2, n3}, strides{n2 * n3, n3} {}
  SERAC_HOST_DEVICE size_t index(size_t i, size_t j, size_t k) const { return i * strides[0] + j * strides[1] + k; }
  size_t                   size(int i = 0) { return sizes[i]; }
};

template <typename T, ExecutionSpace exec>
struct ArrayBase {
  ArrayBase() : ptr{nullptr}, n(0) {}

  ArrayBase(size_t size) : ptr{nullptr}, n(0) { resize(size); }

  ArrayBase(const ArrayBase<T, exec>& arr) : ptr{nullptr}, n(0)
  {
    resize(arr.n);
    detail::copy<exec, exec>(arr.ptr, arr.ptr + arr.n, ptr);
  };

  template <ExecutionSpace other_exec>
  ArrayBase(const ArrayBase<T, other_exec>& arr) : ptr{nullptr}, n(0)
  {
    resize(arr.n);
    detail::copy<other_exec, exec>(arr.ptr, arr.ptr + arr.n, ptr);
  }

  void operator=(const ArrayBase<T, exec>& arr)
  {
    resize(arr.n);
    detail::copy<exec, exec>(arr.ptr, arr.ptr + arr.n, ptr);
  }

  template <ExecutionSpace other_exec>
  void operator=(const ArrayBase<T, other_exec>& arr)
  {
    resize(arr.n);
    detail::copy<other_exec, exec>(arr.ptr, arr.ptr + arr.n, ptr);
  }

  void resize(size_t new_size)
  {
    if (ptr && n != new_size) {
      detail::deallocate<exec>(ptr);
    }
    n   = new_size;
    ptr = detail::allocate<T, exec>(n);
  };

  ~ArrayBase()
  {
    if (ptr) {
      detail::deallocate<exec>(ptr);
    }
  }

  T*     ptr;
  size_t n;
};

}  // namespace detail

template <typename T, size_t d, ExecutionSpace exec = ExecutionSpace::CPU>
struct Array;

template <typename T>
struct Array<T, 1, ExecutionSpace::CPU> : public detail::ArrayBase<T, ExecutionSpace::CPU>,
                                          public detail::Indexable<1> {
  using detail::ArrayBase<T, ExecutionSpace::CPU>::ptr;
  Array() : detail::ArrayBase<T, ExecutionSpace::CPU>(0), detail::Indexable<1>(0) {}
  Array(size_t n) : detail::ArrayBase<T, ExecutionSpace::CPU>(n), detail::Indexable<1>{n} {}

#if defined(__CUDACC__)
  Array(const Array<T, 1, ExecutionSpace::GPU>& other)
      : detail::ArrayBase<T, ExecutionSpace::CPU>(other), detail::Indexable<1>{other}
  {
  }
  void operator=(const Array<T, 1, ExecutionSpace::GPU>& other)
  {
    detail::ArrayBase<T, ExecutionSpace::CPU>::operator=(other);
    detail::Indexable<1>::                     operator=(other);
  }
#endif

  T&       operator()(size_t i) { return ptr[i]; }
  const T& operator()(size_t i) const { return ptr[i]; }
};

template <typename T>
struct Array<T, 2, ExecutionSpace::CPU> : public detail::ArrayBase<T, ExecutionSpace::CPU>,
                                          public detail::Indexable<2> {
  using detail::ArrayBase<T, ExecutionSpace::CPU>::ptr;
  Array() : detail::ArrayBase<T, ExecutionSpace::CPU>(0), detail::Indexable<2>(0, 0) {}
  Array(size_t n1, size_t n2) : detail::ArrayBase<T, ExecutionSpace::CPU>(n1 * n2), detail::Indexable<2>(n1, n2) {}
  Array(const Array<T, 2, ExecutionSpace::CPU>& other)
      : detail::ArrayBase<T, ExecutionSpace::CPU>(other), detail::Indexable<2>{other}
  {
  }

  void operator=(const Array<T, 2, ExecutionSpace::CPU>& other)
  {
    detail::ArrayBase<T, ExecutionSpace::CPU>::operator=(other);
    detail::Indexable<2>::                     operator=(other);
  }

#if defined(__CUDACC__)
  Array(const Array<T, 2, ExecutionSpace::GPU>& other)
      : detail::ArrayBase<T, ExecutionSpace::CPU>(other), detail::Indexable<2>{other}
  {
  }
  void operator=(const Array<T, 2, ExecutionSpace::GPU>& other)
  {
    detail::ArrayBase<T, ExecutionSpace::CPU>::operator=(other);
    detail::Indexable<2>::                     operator=(other);
  }
#endif

  T&       operator()(size_t i, size_t j) { return ptr[index(i, j)]; }
  const T& operator()(size_t i, size_t j) const { return ptr[index(i, j)]; }
};

template <typename T>
struct Array<T, 3, ExecutionSpace::CPU> : public detail::ArrayBase<T, ExecutionSpace::CPU>,
                                          public detail::Indexable<3> {
  using detail::ArrayBase<T, ExecutionSpace::CPU>::ptr;
  Array() : detail::ArrayBase<T, ExecutionSpace::CPU>(0), detail::Indexable<3>(0, 0, 0) {}
  Array(size_t n1, size_t n2, size_t n3)
      : detail::ArrayBase<T, ExecutionSpace::CPU>(n1 * n2 * n3), detail::Indexable<3>(n1, n2, n3)
  {
  }

  Array(const Array<T, 3, ExecutionSpace::CPU>& other)
      : detail::ArrayBase<T, ExecutionSpace::CPU>(other), detail::Indexable<3>{other}
  {
  }

  void operator=(const Array<T, 3, ExecutionSpace::CPU>& other)
  {
    detail::ArrayBase<T, ExecutionSpace::CPU>::operator=(other);
    detail::Indexable<3>::                     operator=(other);
  }

#if defined(__CUDACC__)
  Array(const Array<T, 3, ExecutionSpace::GPU>& other)
      : detail::ArrayBase<T, ExecutionSpace::CPU>(other), detail::Indexable<3>{other}
  {
  }
  void operator=(const Array<T, 3, ExecutionSpace::GPU>& other)
  {
    detail::ArrayBase<T, ExecutionSpace::CPU>::operator=(other);
    detail::Indexable<3>::                     operator=(other);
  }
#endif

  T&       operator()(size_t i, size_t j, size_t k) { return ptr[index(i, j, k)]; }
  const T& operator()(size_t i, size_t j, size_t k) const { return ptr[index(i, j, k)]; }
};

/// @brief set the contents of an array to zero, byte-wise
template <typename T>
void zero_out(detail::ArrayBase<T, ExecutionSpace::CPU>& arr)
{
  std::memset(arr.ptr, 0, arr.n * sizeof(T));
}

/**
 * @tparam T the type stored in the container
 * @tparam d the dimensionality of the array
 * @tparam exec where the memory lives (CPU, GPU)
 *
 * @brief a container that behaves similar to Array, but does not own its data.
 * ArrayView objects have reference semantics, and copying them is inexpensive.
 */
template <typename T, size_t d, ExecutionSpace exec>
struct ArrayView;

template <typename T>
struct ArrayView<T, 1, ExecutionSpace::CPU> : public detail::Indexable<1> {
  ArrayView(T* p, size_t n) : detail::Indexable<1>(n), ptr{p} {}
  ArrayView(const Array<T, 1, ExecutionSpace::CPU>& arr) : detail::Indexable<1>(arr), ptr{arr.ptr} {}
  T&       operator[](size_t i) { return ptr[i]; }
  const T& operator[](size_t i) const { return ptr[i]; }
  T&       operator()(size_t i) { return ptr[i]; }
  const T& operator()(size_t i) const { return ptr[i]; }
  T*       ptr;
};

template <typename T>
struct ArrayView<T, 2, ExecutionSpace::CPU> : public detail::Indexable<2> {
  ArrayView(T* p, size_t n1, size_t n2) : detail::Indexable<2>(n1, n2), ptr{p} {}
  ArrayView(const Array<T, 2, ExecutionSpace::CPU>& arr) : detail::Indexable<2>(arr), ptr{arr.ptr} {}
  T&       operator()(size_t i, size_t j) { return ptr[index(i, j)]; }
  const T& operator()(size_t i, size_t j) const { return ptr[index(i, j)]; }
  T*       ptr;
};

template <typename T>
struct ArrayView<T, 3, ExecutionSpace::CPU> : public detail::Indexable<3> {
  ArrayView(T* p, size_t n1, size_t n2, size_t n3) : detail::Indexable<3>(n1, n2, n3), ptr{p} {}
  ArrayView(const Array<T, 3, ExecutionSpace::CPU>& arr) : detail::Indexable<3>(arr), ptr{arr.ptr} {}
  T&       operator()(size_t i, size_t j, size_t k) { return ptr[index(i, j, k)]; }
  const T& operator()(size_t i, size_t j, size_t k) const { return ptr[index(i, j, k)]; }
  T*       ptr;
};

/// @brief make a view into an Array of a given shape / space
template <typename T, size_t d, ExecutionSpace exec>
auto view(const Array<T, d, exec>& arr)
{
  return ArrayView<T, d, exec>(arr);
}

/// @brief an alias for arrays that reside in CPU memory
template <typename T, size_t d>
using CPUArray = Array<T, d, ExecutionSpace::CPU>;

/// @brief an alias for arrays that reside in GPU memory
template <typename T, size_t d>
using CPUView = ArrayView<T, d, ExecutionSpace::CPU>;

#if defined(__CUDACC__)
template <typename T>
struct Array<T, 1, ExecutionSpace::GPU> : public detail::ArrayBase<T, ExecutionSpace::GPU>,
                                          public detail::Indexable<1> {
  using detail::ArrayBase<T, ExecutionSpace::GPU>::ptr;
  Array() : detail::ArrayBase<T, ExecutionSpace::GPU>(0), detail::Indexable<1>(0) {}
  Array(size_t n) : detail::ArrayBase<T, ExecutionSpace::GPU>(n), detail::Indexable<1>(n) {}
  SERAC_DEVICE T&    operator()(size_t i) { return ptr[i]; }
  SERAC_DEVICE const T& operator()(size_t i) const { return ptr[i]; }
};

template <typename T>
struct Array<T, 2, ExecutionSpace::GPU> : public detail::ArrayBase<T, ExecutionSpace::GPU>,
                                          public detail::Indexable<2> {
  using detail::ArrayBase<T, ExecutionSpace::GPU>::ptr;
  Array() : detail::ArrayBase<T, ExecutionSpace::GPU>(0), detail::Indexable<2>(0, 0) {}
  Array(size_t n1, size_t n2) : detail::ArrayBase<T, ExecutionSpace::GPU>(n1 * n2), detail::Indexable<2>(n1, n2) {}
  SERAC_DEVICE T&    operator()(size_t i, size_t j) { return ptr[index(i, j)]; }
  SERAC_DEVICE const T& operator()(size_t i, size_t j) const { return ptr[index(i, j)]; }
};

template <typename T>
struct Array<T, 3, ExecutionSpace::GPU> : public detail::ArrayBase<T, ExecutionSpace::GPU>,
                                          public detail::Indexable<3> {
  using detail::ArrayBase<T, ExecutionSpace::GPU>::ptr;
  Array() : detail::ArrayBase<T, ExecutionSpace::GPU>(0), detail::Indexable<3>(0, 0, 0) {}
  Array(size_t n1, size_t n2, size_t n3)
      : detail::ArrayBase<T, ExecutionSpace::GPU>(n1 * n2 * n3), detail::Indexable<3>(n1, n2, n3)
  {
  }
  SERAC_DEVICE T&    operator()(size_t i, size_t j, size_t k) { return ptr[index(i, j, k)]; }
  SERAC_DEVICE const T& operator()(size_t i, size_t j, size_t k) const { return ptr[index(i, j, k)]; }
};

template <typename T>
void zero_out(detail::ArrayBase<T, serac::ExecutionSpace::GPU>& arr)
{
  cudaMemset(arr.ptr, 0, arr.n * sizeof(T));
}

template <typename T>
struct ArrayView<T, 1, ExecutionSpace::GPU> : public detail::Indexable<1> {
  ArrayView(T* p, size_t n) : detail::Indexable<1>(n), ptr{p} {}
  ArrayView(const Array<T, 1, ExecutionSpace::GPU>& arr) : ptr{arr.ptr}, detail::Indexable<1>(arr) {}
  SERAC_DEVICE T&    operator[](size_t i) { return ptr[i]; }
  SERAC_DEVICE const T& operator[](size_t i) const { return ptr[i]; }
  SERAC_DEVICE T&    operator()(size_t i) { return ptr[i]; }
  SERAC_DEVICE const T& operator()(size_t i) const { return ptr[i]; }
  T*                    ptr;
};

template <typename T>
struct ArrayView<T, 2, ExecutionSpace::GPU> : public detail::Indexable<2> {
  ArrayView(T* p, size_t n1, size_t n2) : detail::Indexable<2>(n1, n2), ptr{p} {}
  ArrayView(const Array<T, 2, ExecutionSpace::GPU>& arr) : ptr{arr.ptr}, detail::Indexable<2>(arr) {}
  SERAC_DEVICE T&    operator()(size_t i, size_t j) { return ptr[index(i, j)]; }
  SERAC_DEVICE const T& operator()(size_t i, size_t j) const { return ptr[index(i, j)]; }
  T*                    ptr;
};

template <typename T, size_t d>
using GPUArray = Array<T, d, ExecutionSpace::GPU>;

template <typename T, size_t d>
using GPUView = ArrayView<T, d, ExecutionSpace::GPU>;

template <typename T>
struct Array<T, 1, ExecutionSpace::Dynamic> : public detail::ArrayBase<T, ExecutionSpace::Dynamic>,
                                              public detail::Indexable<1> {
  using detail::ArrayBase<T, ExecutionSpace::Dynamic>::ptr;
  Array() : detail::ArrayBase<T, ExecutionSpace::Dynamic>(0), detail::Indexable<1>(0){};
  Array(size_t n) : detail::ArrayBase<T, ExecutionSpace::Dynamic>(n), detail::Indexable<1>(n) {}
  SERAC_HOST_DEVICE T&    operator()(size_t i) { return ptr[i]; }
  SERAC_HOST_DEVICE const T& operator()(size_t i) const { return ptr[i]; }
  SERAC_HOST_DEVICE T*    data() { return ptr; }
  SERAC_HOST_DEVICE const T* data() const { return ptr; }
  T*                         begin() { return ptr; }
  T*                         end() { return ptr + this->size(); }
};

template <typename T>
using ManagedArray = Array<T, 1, ExecutionSpace::Dynamic>;

#endif

}  // namespace serac

//! @endcond

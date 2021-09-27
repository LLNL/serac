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

#pragma once

#include <stddef.h>

#include <iostream>
#include <algorithm>

#include "serac/infrastructure/accelerator.hpp"

namespace serac {

namespace impl {

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
#endif
}

template <ExecutionSpace exec, typename T>
void deallocate(T* ptr)
{
  if constexpr (exec == ExecutionSpace::CPU) {
    free(ptr);
  }

#if defined(__CUDACC__)
  if constexpr (exec == ExecutionSpace::GPU) {
    cudaFree(ptr);
  }
#endif
}

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

template <size_t d>
struct Indexable;

template <>
struct Indexable<1> {
  size_t sizes;
  Indexable(size_t n1) : sizes{n1} {}
  SERAC_HOST_DEVICE size_t index(size_t i) const { return i; }
  size_t                   size(int i = 0) { return sizes; }
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

template <typename T, ExecutionSpace device>
struct ArrayBase {
  ArrayBase() : ptr{nullptr}, n(0) {}

  ArrayBase(size_t size) : ptr{nullptr}, n(0) { resize(size); }

  ArrayBase(const ArrayBase<T, device>& arr) : ptr{nullptr}, n(0)
  {
    resize(arr.n);
    impl::copy<device, device>(arr.ptr, arr.ptr + arr.n, ptr);
  };

  template <ExecutionSpace other_exec>
  ArrayBase(const ArrayBase<T, other_exec>& arr) : ptr{nullptr}, n(0)
  {
    resize(arr.n);
    impl::copy<other_exec, device>(arr.ptr, arr.ptr + arr.n, ptr);
  }

  template <ExecutionSpace other_exec>
  void operator=(const ArrayBase<T, other_exec>& arr)
  {
    resize(arr.n);
    impl::copy<other_exec, device>(arr.ptr, arr.ptr + arr.n, ptr);
  }

  void resize(size_t new_size)
  {
    if (ptr && n != new_size) {
      impl::deallocate<device>(ptr);
    }
    n   = new_size;
    ptr = impl::allocate<T, device>(n);
  };

  ~ArrayBase()
  {
    if (ptr) {
      impl::deallocate<device>(ptr);
    }
  }

  T*     ptr;
  size_t n;
};

}  // namespace impl

template <typename T, size_t d, ExecutionSpace device = ExecutionSpace::CPU>
struct Array;

template <typename T>
struct Array<T, 1, ExecutionSpace::CPU> : public impl::ArrayBase<T, ExecutionSpace::CPU>, public impl::Indexable<1> {
  using impl::ArrayBase<T, ExecutionSpace::CPU>::ptr;
  Array(size_t n) : impl::ArrayBase<T, ExecutionSpace::CPU>(n), impl::Indexable<1>{} {}

#if defined(__CUDACC__)
  Array(const Array<T, 1, ExecutionSpace::GPU>& other) : impl::ArrayBase<T, ExecutionSpace::CPU>(other), impl::Indexable<1>{other} {}
  void operator=(const Array<T, 1, ExecutionSpace::GPU>& other)
  {
    impl::ArrayBase<T, ExecutionSpace::CPU>::operator=(other);
    impl::Indexable<1>::             operator=(other);
  }
#endif

  T&       operator()(size_t i) { return ptr[i]; }
  const T& operator()(size_t i) const { return ptr[i]; }
};

template <typename T>
struct Array<T, 2, ExecutionSpace::CPU> : public impl::ArrayBase<T, ExecutionSpace::CPU>, public impl::Indexable<2> {
  using impl::ArrayBase<T, ExecutionSpace::CPU>::ptr;
  Array(size_t n1, size_t n2) : impl::ArrayBase<T, ExecutionSpace::CPU>(n1 * n2), impl::Indexable<2>(n1, n2) {}

  Array(const Array<T, 2, ExecutionSpace::CPU>& other) : impl::ArrayBase<T, ExecutionSpace::CPU>(other), impl::Indexable<2>{other} {}

  void operator=(const Array<T, 2, ExecutionSpace::CPU>& other)
  {
    impl::ArrayBase<T, ExecutionSpace::CPU>::operator=(other);
    impl::Indexable<2>::             operator=(other);
  }

#if defined(__CUDACC__)
  Array(const Array<T, 2, ExecutionSpace::GPU>& other) : impl::ArrayBase<T, ExecutionSpace::CPU>(other), impl::Indexable<2>{other} {}
  void operator=(const Array<T, 2, ExecutionSpace::GPU>& other)
  {
    impl::ArrayBase<T, ExecutionSpace::CPU>::operator=(other);
    impl::Indexable<2>::             operator=(other);
  }
#endif

  T&       operator()(size_t i, size_t j) { return ptr[index(i, j)]; }
  const T& operator()(size_t i, size_t j) const { return ptr[index(i, j)]; }
};

template <typename T>
struct Array<T, 3, ExecutionSpace::CPU> : public impl::ArrayBase<T, ExecutionSpace::CPU>, public impl::Indexable<3> {
  using impl::ArrayBase<T, ExecutionSpace::CPU>::ptr;
  Array(size_t n1, size_t n2, size_t n3) : impl::ArrayBase<T, ExecutionSpace::CPU>(n1 * n2 * n3), impl::Indexable<3>(n1, n2, n3)
  {
  }

  Array(const Array<T, 3, ExecutionSpace::CPU>& other) : impl::ArrayBase<T, ExecutionSpace::CPU>(other), impl::Indexable<3>{other} {}

  void operator=(const Array<T, 3, ExecutionSpace::CPU>& other)
  {
    impl::ArrayBase<T, ExecutionSpace::CPU>::operator=(other);
    impl::Indexable<3>::             operator=(other);
  }

#if defined(__CUDACC__)
  Array(const Array<T, 3, ExecutionSpace::GPU>& other) : impl::ArrayBase<T, ExecutionSpace::CPU>(other), impl::Indexable<3>{other} {}
  void operator=(const Array<T, 3, ExecutionSpace::GPU>& other)
  {
    impl::ArrayBase<T, ExecutionSpace::CPU>::operator=(other);
    impl::Indexable<3>::             operator=(other);
  }
#endif

  T&       operator()(size_t i, size_t j, size_t k) { return ptr[index(i, j, k)]; }
  const T& operator()(size_t i, size_t j, size_t k) const { return ptr[index(i, j, k)]; }
};

template <typename T, size_t d, ExecutionSpace device>
struct ArrayView;

template <typename T>
struct ArrayView<T, 1, ExecutionSpace::CPU> : public impl::Indexable<1> {
  template <ExecutionSpace device>
  ArrayView(const Array<T, 3, device>& arr) : impl::Indexable<1>(arr), ptr{arr.ptr}
  {
  }
  T&       operator[](size_t i) { return ptr[i]; }
  const T& operator[](size_t i) const { return ptr[i]; }
  T&       operator()(size_t i) { return ptr[i]; }
  const T& operator()(size_t i) const { return ptr[i]; }
  T*       ptr;
};

template <typename T>
struct ArrayView<T, 2, ExecutionSpace::CPU> : public impl::Indexable<2> {
  ArrayView(const Array<T, 2, ExecutionSpace::CPU>& arr) : impl::Indexable<2>(arr), ptr{arr.ptr} {}
  T&       operator()(size_t i, size_t j) { return ptr[index(i, j)]; }
  const T& operator()(size_t i, size_t j) const { return ptr[index(i, j)]; }
  T*       ptr;
};

template <typename T, size_t d, ExecutionSpace device>
auto view(const Array<T, d, device>& arr)
{
  return ArrayView<T, d, device>(arr);
}

template <typename T, size_t d>
using CPUArray = Array<T, d, ExecutionSpace::CPU>;

template <typename T, size_t d>
using CPUView = ArrayView<T, d, ExecutionSpace::CPU>;

#if defined(__CUDACC__)
template <typename T>
struct Array<T, 1, ExecutionSpace::GPU> : public impl::ArrayBase<T, ExecutionSpace::GPU>, public impl::Indexable<1> {
  using impl::ArrayBase<T, ExecutionSpace::GPU>::ptr;
  Array() = default;
  Array(size_t n) : impl::ArrayBase<T, ExecutionSpace::GPU>(n), impl::Indexable<1>(n) {}
  SERAC_DEVICE T&    operator()(size_t i) { return ptr[i]; }
  SERAC_DEVICE const T& operator()(size_t i) const { return ptr[i]; }
};

template <typename T>
struct Array<T, 2, ExecutionSpace::GPU> : public impl::ArrayBase<T, ExecutionSpace::GPU>, public impl::Indexable<2> {
  using impl::ArrayBase<T, ExecutionSpace::GPU>::ptr;
  Array() = default;
  Array(size_t n1, size_t n2) : impl::ArrayBase<T, ExecutionSpace::GPU>(n1 * n2), impl::Indexable<2>(n1, n2) {}
  SERAC_DEVICE T&    operator()(size_t i, size_t j) { return ptr[index(i, j)]; }
  SERAC_DEVICE const T& operator()(size_t i, size_t j) const { return ptr[index(i, j)]; }
};

template <typename T>
struct Array<T, 3, ExecutionSpace::GPU> : public impl::ArrayBase<T, ExecutionSpace::GPU>, public impl::Indexable<3> {
  using impl::ArrayBase<T, ExecutionSpace::GPU>::ptr;
  Array() = default;
  Array(size_t n1, size_t n2, size_t n3) : impl::ArrayBase<T, ExecutionSpace::GPU>(n1 * n2 * n3), impl::Indexable<3>(n1, n2, n3)
  {
  }
  SERAC_DEVICE T&    operator()(size_t i, size_t j, size_t k) { return ptr[index(i, j, k)]; }
  SERAC_DEVICE const T& operator()(size_t i, size_t j, size_t k) const { return ptr[index(i, j, k)]; }
};

template <typename T>
struct ArrayView<T, 1, ExecutionSpace::GPU> : public impl::Indexable<1> {
  template <ExecutionSpace device>
  ArrayView(const Array<T, 1, device>& arr) : ptr{arr.ptr}, impl::Indexable<1>(arr)
  {
  }
  SERAC_DEVICE T&    operator[](size_t i) { return ptr[i]; }
  SERAC_DEVICE const T& operator[](size_t i) const { return ptr[i]; }
  SERAC_DEVICE T&    operator()(size_t i) { return ptr[i]; }
  SERAC_DEVICE const T& operator()(size_t i) const { return ptr[i]; }
  T*                    ptr;
};

template <typename T>
struct ArrayView<T, 2, ExecutionSpace::GPU> : public impl::Indexable<2> {
  template <ExecutionSpace device>
  ArrayView(const Array<T, 2, device>& arr) : ptr{arr.ptr}, impl::Indexable<2>(arr)
  {
  }
  SERAC_DEVICE T&    operator()(size_t i, size_t j) { return ptr[index(i, j)]; }
  SERAC_DEVICE const T& operator()(size_t i, size_t j) const { return ptr[index(i, j)]; }
  T*                    ptr;
};

template <typename T, size_t d>
using GPUArray = Array<T, d, ExecutionSpace::GPU>;

template <typename T, size_t d>
using GPUView = ArrayView<T, d, ExecutionSpace::GPU>;
#endif

}  // namespace serac

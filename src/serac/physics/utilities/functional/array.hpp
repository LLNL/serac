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

#if defined(__CUDACC__)
#define HOST_DEVICE __host__ __device__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST_DEVICE
#define HOST
#define DEVICE
#endif

enum class Device
{
  CPU,
  GPU
};

namespace axom {

namespace impl {

template <typename T, Device device>
T* allocate(size_t n)
{
  if constexpr (device == Device::CPU) {
    return static_cast<T*>(malloc(n * sizeof(T)));
  }

#if defined(__CUDACC__)
  if constexpr (device == Device::GPU) {
    T* ptr;
    cudaMalloc(&ptr, sizeof(T) * n);
    return ptr;
  }
#endif
}

template <Device device, typename T>
void deallocate(T* ptr)
{
  if constexpr (device == Device::CPU) {
    free(ptr);
  }

#if defined(__CUDACC__)
  if constexpr (device == Device::GPU) {
    cudaFree(ptr);
  }
#endif
}

template <Device src_device, Device dst_device, typename T>
void copy(T* src_begin, T* src_end, T* dst_begin)
{
  if constexpr (src_device == Device::CPU && dst_device == Device::CPU) {
    std::copy(src_begin, src_end, dst_begin);
  }

#if defined(__CUDACC__)
  int num_bytes = (src_end - src_begin) * sizeof(T);

  if constexpr (src_device == Device::CPU && dst_device == Device::GPU) {
    cudaMemcpy(dst_begin, src_begin, num_bytes, cudaMemcpyHostToDevice);
  }

  if constexpr (src_device == Device::GPU && dst_device == Device::CPU) {
    cudaMemcpy(dst_begin, src_begin, num_bytes, cudaMemcpyDeviceToHost);
  }

  if constexpr (src_device == Device::GPU && dst_device == Device::GPU) {
    cudaMemcpy(dst_begin, src_begin, num_bytes, cudaMemcpyDeviceToDevice);
  }
#endif
}

template <int d>
struct Indexable;

template <>
struct Indexable<1> {
  HOST_DEVICE int index(int i) const { return i; }
};

template <>
struct Indexable<2> {
  int             stride;
  HOST_DEVICE int index(int i, int j) const { return i * stride + j; }
};

template <>
struct Indexable<3> {
  int             strides[2];
  HOST_DEVICE int index(int i, int j, int k) const { return i * strides[0] + j * strides[1] + k; }
};

template <typename T, Device device>
struct ArrayBase {
  ArrayBase() : ptr{nullptr}, n(0) {}

  ArrayBase(int size) : ptr{nullptr}, n(0) { resize(size); }

  template <Device other_device>
  ArrayBase(const ArrayBase<T, other_device>& arr) : ptr{nullptr}, n(0)
  {
    resize(arr.n);
    impl::copy<other_device, device>(arr.ptr, arr.ptr + arr.n, ptr);
  }

  template <Device other_device>
  void operator=(const ArrayBase<T, other_device>& arr)
  {
    resize(arr.n);
    impl::copy<other_device, device>(arr.ptr, arr.ptr + arr.n, ptr);
  }

  void resize(size_t new_size)
  {
    if (ptr && n != new_size) {
      impl::deallocate<device>(ptr);
    }
    n   = new_size;
    ptr = allocate<T, device>(n);
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

template <typename T, int d, Device device = Device::CPU>
struct Array;

template <typename T>
struct Array<T, 1, Device::CPU> : public impl::ArrayBase<T, Device::CPU>, public impl::Indexable<1> {
  using impl::ArrayBase<T, Device::CPU>::ptr;
  Array() = default;
  Array(int n) : impl::ArrayBase<T, Device::CPU>(n), impl::Indexable<1>{} {}

  #if defined(__CUDACC__)
  Array(const Array<T,1,Device::GPU> & other) : impl::ArrayBase<T, Device::CPU>(other), impl::Indexable<1>{other} {}
  void operator=(const Array<T,1,Device::GPU> & other) {
    impl::ArrayBase<T, Device::CPU>::operator=(other);
    impl::Indexable<1>::operator=(other);
  } 
  #endif

  T&       operator()(int i) { return ptr[i]; }
  const T& operator()(int i) const { return ptr[i]; }
};

template <typename T>
struct Array<T, 2, Device::CPU> : public impl::ArrayBase<T, Device::CPU>, public impl::Indexable<2> {
  using impl::ArrayBase<T, Device::CPU>::ptr;
  Array() = default;
  Array(int n1, int n2) : impl::ArrayBase<T, Device::CPU>(n1 * n2), impl::Indexable<2>{n2} {}

  #if defined(__CUDACC__)
  Array(const Array<T,2,Device::GPU> & other) : impl::ArrayBase<T, Device::CPU>(other), impl::Indexable<2>{other} {}
  void operator=(const Array<T,2,Device::GPU> & other) {
    impl::ArrayBase<T, Device::CPU>::operator=(other);
    impl::Indexable<2>::operator=(other);
  } 
  #endif

  T&       operator()(int i, int j) { return ptr[index(i, j)]; }
  const T& operator()(int i, int j) const { return ptr[index(i, j)]; }
};

template <typename T>
struct Array<T, 3, Device::CPU> : public impl::ArrayBase<T, Device::CPU>, public impl::Indexable<3> {
  using impl::ArrayBase<T, Device::CPU>::ptr;
  Array() = default;
  Array(int n1, int n2, int n3) : impl::ArrayBase<T, Device::CPU>(n1 * n2 * n3), impl::Indexable<3>{n2 * n3, n3} {}
  T&       operator()(int i, int j, int k) { return ptr[index(i, j, k)]; }
  const T& operator()(int i, int j, int k) const { return ptr[index(i, j, k)]; }
};

template <typename T, int d, Device device>
struct ArrayView;

template <typename T>
struct ArrayView<T, 1, Device::CPU> : public impl::Indexable<1> {
  template <Device device>
  ArrayView(const Array<T, 3, device>& arr) : impl::Indexable<1>(arr), ptr{arr.ptr} {}
  T&       operator[](int i) { return ptr[i]; }
  const T& operator[](int i) const { return ptr[i]; }
  T&       operator()(int i) { return ptr[i]; }
  const T& operator()(int i) const { return ptr[i]; }
  T*       ptr;
};

template <typename T>
struct ArrayView<T, 2, Device::CPU> : public impl::Indexable<2> {
  ArrayView(const Array<T, 2, Device::CPU>& arr) : impl::Indexable<2>(arr), ptr{arr.ptr} {}
  T&       operator()(int i, int j) { return ptr[index(i, j)]; }
  const T& operator()(int i, int j) const { return ptr[index(i, j)]; }
  T*       ptr;
};

template <typename T, int d, Device device>
auto view(const Array<T, d, device>& arr)
{
  return ArrayView<T, d, device>(arr);
}

template <typename T, int d>
using CPUArray = Array<T, d, Device::CPU>;

template <typename T, int d>
using CPUView = ArrayView<T, d, Device::CPU>;

#if defined(__CUDACC__)
template <typename T>
struct Array<T, 1, Device::GPU> : public impl::ArrayBase<T, Device::GPU>, public impl::Indexable<1> {
  using impl::ArrayBase<T, Device::GPU>::ptr;
  Array() = default;
  Array(int n) : impl::ArrayBase<T, Device::GPU>(n), impl::Indexable<1>{} {}
  DEVICE T&    operator()(int i) { return ptr[i]; }
  DEVICE const T& operator()(int i) const { return ptr[i]; }
};

template <typename T>
struct Array<T, 2, Device::GPU> : public impl::ArrayBase<T, Device::GPU>, public impl::Indexable<2> {
  using impl::ArrayBase<T, Device::GPU>::ptr;
  Array() = default;
  Array(int n1, int n2) : impl::ArrayBase<T, Device::GPU>(n1 * n2), impl::Indexable<2>{n2} {}
  DEVICE T&    operator()(int i, int j) { return ptr[index(i, j)]; }
  DEVICE const T& operator()(int i, int j) const { return ptr[index(i, j)]; }
};

template <typename T>
struct Array<T, 3, Device::GPU> : public impl::ArrayBase<T, Device::GPU>, public impl::Indexable<3> {
  using impl::ArrayBase<T, Device::GPU>::ptr;
  Array() = default;
  Array(int n1, int n2, int n3) : impl::ArrayBase<T, Device::GPU>(n1 * n2 * n3), impl::Indexable<3>{n2 * n3, n3} {}
  DEVICE T&    operator()(int i, int j, int k) { return ptr[index(i, j, k)]; }
  DEVICE const T& operator()(int i, int j, int k) const { return ptr[index(i, j, k)]; }
};

template <typename T>
struct ArrayView<T, 1, Device::GPU> : public impl::Indexable<1> {
  template <Device device>
  ArrayView(const Array<T, 1, device>& arr) : ptr{arr.ptr}, impl::Indexable<1>(arr)
  {
  }
  DEVICE T&    operator[](int i) { return ptr[i]; }
  DEVICE const T& operator[](int i) const { return ptr[i]; }
  DEVICE T&    operator()(int i) { return ptr[i]; }
  DEVICE const T& operator()(int i) const { return ptr[i]; }
  T*              ptr;
};

template <typename T>
struct ArrayView<T, 2, Device::GPU> : public impl::Indexable<2> {
  template <Device device>
  ArrayView(const Array<T, 2, device>& arr) : ptr{arr.ptr}, impl::Indexable<2>(arr)
  {
  }
  DEVICE T&    operator()(int i, int j) { return ptr[index(i, j)]; }
  DEVICE const T& operator()(int i, int j) const { return ptr[index(i, j)]; }
  T*              ptr;
};

template <typename T, int d>
using GPUArray = Array<T, d, Device::GPU>;

template <typename T, int d>
using GPUView = ArrayView<T, d, Device::GPU>;
#endif

}

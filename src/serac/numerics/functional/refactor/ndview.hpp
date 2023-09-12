#pragma once

#include <utility>

template < typename T, uint32_t rank = 1 >
struct ndview{

  template < auto i >
  static auto Iseq() {
    return std::make_integer_sequence< decltype(i), i >();
  }

  ndview() {}

  ndview(T * input, const std::array<uint32_t, rank> & dimensions) {
    data = input;
    for (uint32_t i = 0; i < rank; i++) {
      uint32_t id = rank - 1 - i;
      shape[id] = dimensions[id];
      strides[id] = (id == rank - 1) ? 1 : strides[id+1] * shape[id+1];
    }
  }

  ndview(T * input, const uint32_t (& dimensions)[rank], const uint32_t (& custom_strides)[rank]) {
    data = input;
    for (uint32_t i = 0; i < rank; i++) {
      shape[i] = dimensions[i];
      strides[i] = custom_strides[i];
    }
  }

  template < typename ... index_types >
  decltype(auto) operator()(index_types ... indices) { 
    if constexpr (sizeof ... (indices) < rank) {
      constexpr uint32_t m = sizeof ... (indices);
      std::array<uint32_t, rank - m> slice_shape;
      for (uint32_t j = 0; j < rank - m; j++) {
        slice_shape[j] = shape[j + m];
      }
      return ndview< T, rank - m >(data + index(Iseq<m>(), indices...), slice_shape);
    } else {
      static_assert(sizeof ... (indices) == rank);
      return static_cast<T &>(data[index(Iseq<rank>(), indices...)]);
    }
  }

  template < typename ... index_types >
  decltype(auto) operator()(index_types ... indices) const { 
    if constexpr (sizeof ... (indices) < rank) {
      constexpr uint32_t m = sizeof ... (indices);
      std::array<uint32_t, rank - m> slice_shape;
      for (int j = 0; j < rank - m; j++) {
        slice_shape[j] = shape[j + m];
      }
      return ndview< T, rank - m >(data + index(Iseq<m>(), indices...), slice_shape);
    } else {
      static_assert(sizeof ... (indices) == rank);
      return static_cast<T &>(data[index(Iseq<rank>(), indices...)]);
    }
  }

  template < uint32_t ... I, typename ... index_types >
  auto index(std::integer_sequence<uint32_t, I...>, index_types ... indices) const {
    return ((uint32_t(indices) * strides[I]) + ...);
  }

  T * data;
  uint32_t shape[rank];
  uint32_t strides[rank];
};

template <int i1, int i2>
auto contract(const ndview<double, 2>& A, const ndview<double, 2>& B, ndview<double, 2> C)
{
  uint32_t d1 = C.shape[0];
  uint32_t d2 = C.shape[1];
  uint32_t d3 = A.shape[i1];


  assert(A.shape[1-i1] == d1);
  assert(B.shape[1-i2] == d2);
  assert(B.shape[i1] == d3);

  for (uint32_t i = 0; i < d1; i++) {
    for (uint32_t j = 0; j < d2; j++) {
      double sum{};
      for (uint32_t k = 0; k < d3; k++) {
        if constexpr (i1 == 0 && i2 == 0) sum += A(k, j) * B(k, i);
        if constexpr (i1 == 1 && i2 == 0) sum += A(i, k) * B(k, j);
        if constexpr (i1 == 0 && i2 == 1) sum += A(k, j) * B(i, k);
        if constexpr (i1 == 1 && i2 == 1) sum += A(i, k) * B(j, k);
      }
      C(i, j) = sum;
    }
  }

  return C;
}

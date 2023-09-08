#pragma once

#include <utility>

template < typename T, int rank = 1 >
struct ndview{
  static constexpr auto iseq = std::make_integer_sequence<int, rank>{};

  ndview() {}

  ndview(T * input, const int (& dimensions)[rank]) {
    data = input;
    for (int i = 0; i < rank; i++) {
      int id = rank - 1 - i;
      shape[id] = dimensions[id];
      strides[id] = (id == rank - 1) ? 1 : strides[id+1] * shape[id+1];
    }
  }

  template < typename ... index_types >
  auto & operator()(index_types ... indices) { 
    static_assert(sizeof ... (indices) == rank);
    return data[index(iseq, indices...)];
  }

  template < typename ... index_types >
  auto & operator()(index_types ... indices) const { 
    static_assert(sizeof ... (indices) == rank);
    return data[index(iseq, indices...)];
  }

  template < int ... I, typename ... index_types >
  auto index(std::integer_sequence<int, I...>, index_types ... indices) const {
    return ((indices * strides[I]) + ...);
  }

  T * data;
  int shape[rank];
  int strides[rank];
};
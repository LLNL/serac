#pragma once

template < int ... n >
struct Dimensions{
  constexpr int operator[](int i) {
    if constexpr (sizeof ...(n) == 0) {
      return 0;
    } else {
      constexpr int values[sizeof...(n)] = {n ...};
      return values[i]; 
    }
  }
};

namespace impl {

  template < int I, int ... n >
  constexpr auto get(std::integer_sequence<int, n...>) {
    constexpr int values[sizeof...(n)] = {n ...};
    return values[I]; 
  }

  template < int r, int ... n, int ... i >
  constexpr auto remove_helper(Dimensions<n...>, std::integer_sequence<int,i...>) {
    return Dimensions<get<i+(i>=r)>(std::integer_sequence<int,n ...>{}) ... >{};
  }

}

template < int m, int ... n >
auto remove_first(Dimensions< m, n... >) { return Dimensions<n ... >{}; }

template < int ... n >
auto remove_last(Dimensions< n... > dim) {
  return impl::remove_helper<(sizeof...(n)) - 1>(dim, std::make_integer_sequence< int, (sizeof ... (n)) - 1 >{});
}

template < int ... m, int ... n >
auto concatenate(Dimensions< m ... >, Dimensions<n...>) { return Dimensions< m..., n ... >{}; };

template < int ... m, int ... n >
auto operator+(Dimensions< m ... >, Dimensions<n...>) { return Dimensions< m..., n ... >{}; };

template < int m, int ... n >
constexpr auto first(Dimensions< m, n... >) { return m; }
constexpr auto first(Dimensions<>) { return 0; }

template < int ... n >
constexpr auto last(Dimensions< n... > A) { return A[sizeof ... (n) - 1]; }
constexpr auto last(Dimensions<>) { return 0; }


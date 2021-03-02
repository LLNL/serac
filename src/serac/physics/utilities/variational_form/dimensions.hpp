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

  template < typename T >
  struct first;

  template < int m, int ... n >
  struct first< Dimensions<m, n...> >{
    using type = Dimensions<m>;
    static constexpr int value = m;
  };

  template < int r, int ... n, int ... i >
  constexpr auto remove_helper(Dimensions<n...>, std::integer_sequence<int,i...>) {
    return Dimensions<get<i+(i>=r)>(std::integer_sequence<int,n ...>{}) ... >{};
  }

  template < typename T >
  struct remove_first;

  template < int m, int ... n >
  struct remove_first< Dimensions<m, n...> >{
    using type = Dimensions< n ... >;
  };

  template < typename T >
  struct remove_last;

  template < int ... n >
  struct remove_last< Dimensions<n...> >{
    using type = decltype(remove_helper<(sizeof...(n)) - 1>(Dimensions<n...>{}, std::make_integer_sequence< int, (sizeof ... (n)) - 1 >{}));
  };

  template < typename S, typename T >
  struct concatenate;

  template < int ... m, int ... n >
  struct concatenate< Dimensions< m ... >, Dimensions<n...> >{
    using type = Dimensions< m..., n ... >;
  };

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


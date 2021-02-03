#pragma once

#include <iostream>
#include <tuple>

template <typename T, int... n>
struct array;

template <typename T, int first, int... rest>
struct array<T, first, rest...> {
  static constexpr int rank = 1 + sizeof...(rest);
  static constexpr int leading_dimension = first;
  using type = typename array<T, rest...>::type[first];
  using slice_type = array<T, rest...>;
  template <int i>
  constexpr auto get() const {
    return (*this)[i];
  }
  template <int i>
  constexpr auto& get() {
    return (*this)[i];
  }
  constexpr auto& operator[](int i) { return values[i]; };
  constexpr auto operator[](int i) const { return values[i]; };
  constexpr auto begin() const { return &values[0]; }
  constexpr auto end() const { return &values[first]; }
  constexpr auto size() const { return first; }
  slice_type values[first];
};

template <typename T, int n>
struct array<T, n> {
  static constexpr int rank = 1;
  static constexpr int leading_dimension = n;
  using type = T[n];
  using slice_type = T;
  template <int i>
  constexpr auto get() const {
    return (*this)[i];
  }
  template <int i>
  constexpr auto& get() {
    return (*this)[i];
  }
  constexpr auto& operator[](int i) { return values[i]; };
  constexpr auto operator[](int i) const { return values[i]; };
  constexpr auto begin() const { return &values[0]; }
  constexpr auto end() const { return &values[n]; }
  constexpr auto size() const { return n; }
  slice_type values[n];
};

template < typename T, int n1 >
array(const T (& data)[n1]) -> array<T, n1>;

template < typename T, int n1, int n2 >
array(const T (& data)[n1][n2]) -> array<T, n1, n2>;

template <typename T, int n>
constexpr auto reverse(const array<T, n>& original) {
  array<T, n> reversed{};
  for (int i = 0; i < n; i++) {
    reversed[i] = original[n - 1 - i];
  }
  return reversed;
}

template <typename T, int n>
constexpr auto rotate(const array<T, n>& original, int rotation) {
  array<T, n> rotated{};
  for (int i = 0; i < n; i++) {
    rotated[i] = original[(i + rotation % n + n) % n];
  }
  return rotated;
}

constexpr int orientation(const array<int, 2> a, const array<int, 2> b) { return (a[0] == b[0]) ? 1 : -1; }

template <int n>
constexpr auto orientation(const array<int, n> a, const array<int, n> b) {
  for (int i = 0; i < n; i++) {
    if (a[0] == b[i]) { 
      return (a[1] == b[(i+1)%n]) ? std::tuple{1, i} : std::tuple{-1, n-1-i}; 
    }
  }
  return std::tuple{0, 0};
}

template <typename T, int n>
constexpr auto total(const array<T, n> a) {
  T p{0};
  for (int i = 0; i < n; i++) {
    p += a[i];
  }
  return p;
}

template <typename T, int n>
constexpr auto product(const array<T, n> a) {
  T p{1};
  for (int i = 0; i < n; i++) {
    p *= a[i];
  }
  return p;
}

template <int n>
constexpr auto any(const array<bool, n> a) {
  for (int i = 0; i < n; i++) {
    if (a[i]) {
      return true;
    }
  }
  return false;
}

template <int n>
constexpr auto all(const array<bool, n> a) {
  for (int i = 0; i < n; i++) {
    if (!a[i]) {
      return false;
    }
  }
  return true;
}

#define binary_operator_overload(x)                                \
  template <typename T, int... n>                                  \
  constexpr auto operator x(const array<T, n...>& a,               \
                            const array<T, n...>& b) {             \
    using S = decltype(T {} x T{});                                \
    array<S, n...> c{};                                            \
    for (int i = 0; i < (n * ...); i++) {                          \
      static_cast<S*>(c.values)[i] = static_cast<T*>(a.values)[i] x(static_cast<T*>(b.values)[i]); \
    }                                                              \
    return c;                                                      \
  }                                                                \
  template <typename T, int... n>                                  \
  constexpr auto operator x(const array<T, n...>& a, const T& b) { \
    using S = decltype(T {} x T{});                                \
    array<S, n...> c{};                                            \
    for (int i = 0; i < (n * ...); i++) {                          \
      static_cast<S*>(c.values)[i] = static_cast<T*>(a.values)[i] x b;	\
    }                                                              \
    return c;                                                      \
  }                                                                \
  template <typename T, int... n>                                  \
  constexpr auto operator x(const T& a, const array<T, n...>& b) { \
    using S = decltype(T {} x T{});                                \
    array<S, n...> c{};                                            \
    for (int i = 0; i < (n * ...); i++) {                          \
      static_cast<S*>(c.values)[i] = a x(static_cast<T*>(b.values))[i];	\
    }                                                              \
    return c;                                                      \
  }

binary_operator_overload(+);
binary_operator_overload(-);
binary_operator_overload(/);
binary_operator_overload(*);
binary_operator_overload(<);
binary_operator_overload(>);
binary_operator_overload(<=);
binary_operator_overload(>=);
binary_operator_overload(==);
binary_operator_overload(!=);

#undef binary_operator_overload

#define compound_assignment_operator_overload(x)                               \
  template <typename T, int... n>                                              \
  constexpr auto& operator x(array<T, n...>& arr, const T& rhs) {              \
    for (int i = 0; i < (n * ...); i++) {                                      \
      static_cast<T*>(arr.values)[i] x rhs;				\
    }                                                                          \
    return arr;                                                                \
  }                                                                            \
  template <typename T, int... n>                                              \
  constexpr auto& operator x(array<T, n...>& arr, const array<T, n...>& rhs) { \
    for (int i = 0; i < (n * ...); i++) {                                      \
      static_cast<T*>(arr.values)[i] x(static_cast<T*>(rhs.values))[i];	\
    }                                                                          \
    return arr;                                                                \
  }

compound_assignment_operator_overload(+=);
compound_assignment_operator_overload(-=);
compound_assignment_operator_overload(*=);
compound_assignment_operator_overload(/=);

#undef compound_assignment_operator_overload

template <typename T, int... n>
auto& operator<<(std::ostream& out, array<T, n...> arr) {
  out << '{' << arr[0];
  for (int i = 1; i < array<T, n...>::leading_dimension; i++) {
    out << ", " << arr[i];
  }
  out << '}';
  return out;
}

template < int I, int n, typename T >
constexpr auto remove(array<T, n> values) {
  static_assert(I < n, "error: trying to remove element beyond array extent");
  array<T,n-1> removed{};
  for (int i = 0; i < n-1; i++) {
    removed[i] = values[i + (I<=i)]; 
  }
  return removed;
}

template < typename T, int n1, int n2 >
constexpr auto join(array<T, n1> v1, array<T,n2> v2) {
  array<T,n1+n2> joined{};
  for (int i = 0; i < n1; i++) { joined[i] = v1[i]; }
  for (int i = 0; i < n2; i++) { joined[i+n1] = v2[i]; }
  return joined;
}

template <int... n>
struct IndexSpace {
  static constexpr int size = sizeof...(n);
  array<char, size> i;
  constexpr auto begin() { return IndexSpace<n...>{}; }
  constexpr auto end() {
    constexpr char dim[size] = {n...};
    IndexSpace<n...> r{};
    r.i[0] = dim[0];
    return r;
  }
  constexpr bool operator!=(IndexSpace<n...> o) { return any(i != o.i); }
  constexpr auto operator*() { return i; }
  constexpr void operator++() {
    constexpr char dim[size] = {n...};
    i[size - 1]++;
    for (int j = size - 1; j > 0; j--) {
      if (i[j] == dim[j]) {
        i[j] = 0;
        i[j - 1]++;
      }
    }
  }
};

template <typename T, int... n>
constexpr auto indices(array<T, n...>) { return IndexSpace<n...>{}; }

// these are required for enabling structured binding accesses like
// auto [a,b,c] = array<int,3>{};
namespace std {
template <typename T, int... n>
struct tuple_size<::array<T, n...>> {
  static constexpr int value = ::array<T, n...>::leading_dimension;
};

template <size_t i, typename T, int... n>
struct tuple_element<i, ::array<T, n...>> {
  using type = typename ::array<T, n...>::slice_type;
};
}  // namespace std

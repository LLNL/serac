// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file array.hpp
 *
 * @brief This file contains the declaration of a multidimensional array class
 */

#pragma once

#include <iostream>
#include <tuple>

namespace serac {

/// @cond
template <typename T, int... n>
struct array;
/// @endcond

/**
 * @brief Multidimensional array class
 * @tparam T The scalar type of the array element
 */
template <typename T, int first, int... rest>
struct array<T, first, rest...> {
  /**
   * @brief The tensor rank of the array (number of indexing dimensions)
   */
  static constexpr int rank              = 1 + sizeof...(rest);
  static constexpr int leading_dimension = first;
  /**
   * @brief The raw array type, e.g., @p T[first][second][third]
   */
  using type = typename array<T, rest...>::type[first];
  /**
   * @brief The type of each "row" of the array
   */
  using slice_type = array<T, rest...>;
  /**
   * @brief Returns the @a i th element of the array
   * @tparam i The index to retrieve
   */
  template <int i>
  constexpr auto get() const
  {
    return (*this)[i];
  }
  /// @overload
  template <int i>
  constexpr auto& get()
  {
    return (*this)[i];
  }
  /**
   * @brief Returns the @a i th element of the array
   * @param[in] i The index to retrieve
   */
  constexpr auto& operator[](int i) { return values[i]; };
  /// @overload
  constexpr auto operator[](int i) const { return values[i]; };
  /**
   * @brief Iterator to the first element of the array
   */
  constexpr auto begin() const { return &values[0]; }
  /**
   * @brief Iterator to the one-past-the-end of the array
   */
  constexpr auto end() const { return &values[first]; }
  /**
   * @brief Returns the leading dimension of the array
   */
  constexpr auto size() const { return first; }
  /**
   * @brief The data in the array, as a static array
   */
  slice_type values[first];
};

/// @cond
template <typename T, int n>
struct array<T, n> {
  static constexpr int rank              = 1;
  static constexpr int leading_dimension = n;
  using type                             = T[n];
  using slice_type                       = T;
  template <int i>
  constexpr auto get() const
  {
    return (*this)[i];
  }
  template <int i>
  constexpr auto& get()
  {
    return (*this)[i];
  }
  constexpr auto& operator[](int i) { return values[i]; };
  constexpr auto  operator[](int i) const { return values[i]; };
  constexpr auto  begin() const { return &values[0]; }
  constexpr auto  end() const { return &values[n]; }
  constexpr auto  size() const { return n; }
  slice_type      values[n];
};
/// @endcond

// Deduction guides for array literals

template <typename T, int n1>
array(const T (&data)[n1]) -> array<T, n1>;

template <typename T, int n1, int n2>
array(const T (&data)[n1][n2]) -> array<T, n1, n2>;

/**
 * @brief Returns the reverse of a one-dimensional array
 * @param[in] original The array to reverse
 */
template <typename T, int n>
constexpr auto reverse(const array<T, n>& original)
{
  array<T, n> reversed{};
  for (int i = 0; i < n; i++) {
    reversed[i] = original[n - 1 - i];
  }
  return reversed;
}

/**
 * @brief Returns a rotated one-dimensional array
 * @param[in] original The array to rotate
 * @param[in] rotation The number of positions to rotate right
 */
template <typename T, int n>
constexpr auto rotate(const array<T, n>& original, int rotation)
{
  array<T, n> rotated{};
  for (int i = 0; i < n; i++) {
    rotated[i] = original[(i + rotation % n + n) % n];
  }
  return rotated;
}

/// @cond
constexpr int orientation(const array<int, 2> a, const array<int, 2> b) { return (a[0] == b[0]) ? 1 : -1; }
/// @endcond

// NOTE: Currently not used anywhere
/**
 * @brief
 */
template <int n>
constexpr auto orientation(const array<int, n>& a, const array<int, n>& b)
{
  for (int i = 0; i < n; i++) {
    if (a[0] == b[i]) {
      return (a[1] == b[(i + 1) % n]) ? std::tuple{1, i} : std::tuple{-1, n - 1 - i};
    }
  }
  return std::tuple{0, 0};
}

/**
 * @brief Computes the sum of all entries in the array
 * @param[in] a The array to sum
 */
template <typename T, int n>
constexpr auto total(const array<T, n>& a)
{
  T p{0};
  for (int i = 0; i < n; i++) {
    p += a[i];
  }
  return p;
}

/**
 * @brief Computes the product of all entries in the array
 * @param[in] a The array to compute on
 */
template <typename T, int n>
constexpr auto product(const array<T, n>& a)
{
  T p{1};
  for (int i = 0; i < n; i++) {
    p *= a[i];
  }
  return p;
}

/**
 * @brief Checks if any elements of a one-dimensional array are @p true
 * @param[in] a The array to search through
 */
template <int n>
constexpr auto any(const array<bool, n>& a)
{
  for (int i = 0; i < n; i++) {
    if (a[i]) {
      return true;
    }
  }
  return false;
}

/**
 * @brief Checks if all elements of a one-dimensional array are @p true
 * @param[in] a The array to search through
 */
template <int n>
constexpr auto all(const array<bool, n>& a)
{
  for (int i = 0; i < n; i++) {
    if (!a[i]) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Generates const + non-const overloads for a binary operator for arrays of equivalent shape
 * The result array is obtained by applying the binary operator to each element of the array
 * @param[in] x The operator to generate overloads for
 */
#define binary_operator_overload(x)                                                                \
  template <typename T, int... n>                                                                  \
  constexpr auto operator x(const array<T, n...>& a, const array<T, n...>& b)                      \
  {                                                                                                \
    using S = decltype(T {} x T{});                                                                \
    array<S, n...> c{};                                                                            \
    for (int i = 0; i < (n * ...); i++) {                                                          \
      static_cast<S*>(c.values)[i] = static_cast<T*>(a.values)[i] x(static_cast<T*>(b.values)[i]); \
    }                                                                                              \
    return c;                                                                                      \
  }                                                                                                \
  template <typename T, int... n>                                                                  \
  constexpr auto operator x(const array<T, n...>& a, const T& b)                                   \
  {                                                                                                \
    using S = decltype(T {} x T{});                                                                \
    array<S, n...> c{};                                                                            \
    for (int i = 0; i < (n * ...); i++) {                                                          \
      static_cast<S*>(c.values)[i] = static_cast<T*>(a.values)[i] x b;                             \
    }                                                                                              \
    return c;                                                                                      \
  }                                                                                                \
  template <typename T, int... n>                                                                  \
  constexpr auto operator x(const T& a, const array<T, n...>& b)                                   \
  {                                                                                                \
    using S = decltype(T {} x T{});                                                                \
    array<S, n...> c{};                                                                            \
    for (int i = 0; i < (n * ...); i++) {                                                          \
      static_cast<S*>(c.values)[i] = a x(static_cast<T*>(b.values))[i];                            \
    }                                                                                              \
    return c;                                                                                      \
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

/**
 * @brief Generates overloads for compound assigment operators
 * Both assignment of an array of equivalent size and assignment of a scalar element are generated
 * @param[in] x The operator to generate overloads for
 */
#define compound_assignment_operator_overload(x)                             \
  template <typename T, int... n>                                            \
  constexpr auto& operator x(array<T, n...>& arr, const T& rhs)              \
  {                                                                          \
    for (int i = 0; i < (n * ...); i++) {                                    \
      static_cast<T*>(arr.values)[i] x rhs;                                  \
    }                                                                        \
    return arr;                                                              \
  }                                                                          \
  template <typename T, int... n>                                            \
  constexpr auto& operator x(array<T, n...>& arr, const array<T, n...>& rhs) \
  {                                                                          \
    for (int i = 0; i < (n * ...); i++) {                                    \
      static_cast<T*>(arr.values)[i] x(static_cast<T*>(rhs.values))[i];      \
    }                                                                        \
    return arr;                                                              \
  }

compound_assignment_operator_overload(+=);
compound_assignment_operator_overload(-=);
compound_assignment_operator_overload(*=);
compound_assignment_operator_overload(/=);

#undef compound_assignment_operator_overload

/**
 * @brief Outputs an array to a stream
 * @param[inout] out The stream to insert into
 * @param[in] arr The array to output
 */
template <typename T, int... n>
auto& operator<<(std::ostream& out, const array<T, n...>& arr)
{
  out << '{' << arr[0];
  for (int i = 1; i < array<T, n...>::leading_dimension; i++) {
    out << ", " << arr[i];
  }
  out << '}';
  return out;
}

/**
 * @brief Removes an element at the specified index
 * @param[in] values The array to remove from
 * @tparam I The index to remove from
 */
template <int I, int n, typename T>
constexpr auto remove(const array<T, n>& values)
{
  static_assert(I < n, "error: trying to remove element beyond array extent");
  array<T, n - 1> removed{};
  for (int i = 0; i < n - 1; i++) {
    removed[i] = values[i + (I <= i)];
  }
  return removed;
}

/**
 * @brief Concatenates two one-dimensional arrays
 * @param[in] v1 The first (left) array
 * @param[in] v2 The second (right) array
 */
template <typename T, int n1, int n2>
constexpr auto join(const array<T, n1>& v1, const array<T, n2>& v2)
{
  array<T, n1 + n2> joined{};
  for (int i = 0; i < n1; i++) {
    joined[i] = v1[i];
  }
  for (int i = 0; i < n2; i++) {
    joined[i + n1] = v2[i];
  }
  return joined;
}

/**
 * @brief Helper type for iterating over a multidimensional array/tensor
 * @tparam n The parameter pack of integer indices
 */
template <int... n>
struct IndexSpace {
  static constexpr int size = sizeof...(n);
  /**
   * @brief The current indices within the array/tensor
   */
  array<char, size> i;  // FIXME: Is char too small here?
  /**
   * @brief Returns a @p begin iterator (first element of the array/tensor)
   */
  constexpr auto begin() { return IndexSpace<n...>{}; }
  /**
   * @brief Returns an @p end iterator (one past the end)
   */
  constexpr auto end()
  {
    constexpr char   dim[size] = {n...};
    IndexSpace<n...> r{};
    // An out-of-bounds index - the index in the leading dimension is set to
    // one past the end
    r.i[0] = dim[0];
    return r;
  }
  /**
   * @brief Checks for inequality of two @p IndexSpaces of equivalent dimension
   */
  constexpr bool operator!=(const IndexSpace<n...>& o) { return any(i != o.i); }
  /**
   * @brief Returns the array of indices
   */
  constexpr auto operator*() { return i; }
  /**
   * @brief Advances the index by one, updating the indices in all dimensions appropriately
   */
  constexpr auto& operator++()
  {
    constexpr char dim[size] = {n...};
    i[size - 1]++;
    for (int j = size - 1; j > 0; j--) {
      if (i[j] == dim[j]) {
        i[j] = 0;
        i[j - 1]++;
      }
    }
    return *this;
  }
};

/**
 * @brief Returns the indices for a given array
 */
template <typename T, int... n>
constexpr auto indices(const array<T, n...>&)
{
  return IndexSpace<n...>{};
}

}  // namespace serac

/**
 * @brief Specializations required for structured bindings like
 * @code{.cpp}
 * auto [a,b,c] = serac::array<int,3>{};
 * @endcode
 */
namespace std {
template <typename T, int... n>
struct tuple_size<serac::array<T, n...>> {
  static constexpr int value = serac::array<T, n...>::leading_dimension;
};

template <size_t i, typename T, int... n>
struct tuple_element<i, serac::array<T, n...>> {
  using type = typename serac::array<T, n...>::slice_type;
};
}  // namespace std

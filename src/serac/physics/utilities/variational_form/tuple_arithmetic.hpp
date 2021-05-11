// this file defines basic arithmetic operations on tuples of values
// so that expressions like sum1 and sum2 below are equivalent
//
// std::tuple< foo, bar > a;
// std::tuple< baz, qux > b;
//
// std::tuple sum1 = a + b;
// std::tuple sum2{std::get<0>(a) + std::get<0>(b), std::get<1>(a) + std::get<1>(b)};

#pragma once

#include <tuple>

#include "tensor.hpp"

namespace serac {

namespace detail {

/**
 * @brief Trait for checking if a type is a @p std::tuple
 */
template <typename T>
struct is_tuple {
  // FIXME: Should we use std::false_type/std::true_type
  static constexpr bool value = false;
};
/// @overload
template <typename... T>
struct is_tuple<std::tuple<T...> > {
  static constexpr bool value = true;
};

/**
 * @brief Trait for checking if a type if a @p std::tuple containting only @p std::tuple
 */
template <typename T>
struct is_tuple_of_tuples {
  static constexpr bool value = false;
};
/// @overload
template <typename... T>
struct is_tuple_of_tuples<std::tuple<T...> > {
  static constexpr bool value = (is_tuple<T>::value && ...);
};

/////////////////////////////////////////////////

// apply operator+ elementwise to entries in two equally-sized tuples
template <typename... S, typename... T, int... I>
constexpr auto plus_helper(const std::tuple<S...>& A, const std::tuple<T...>& B, std::integer_sequence<int, I...>)
{
  return std::make_tuple((std::get<I>(A) + std::get<I>(B))...);
}

// apply operator- elementwise to entries in two equally-sized tuples
template <typename... S, typename... T, int... I>
constexpr auto minus_helper(const std::tuple<S...>& A, const std::tuple<T...>& B, std::integer_sequence<int, I...>)
{
  return std::make_tuple((std::get<I>(A) - std::get<I>(B))...);
}

// apply operator* elementwise to entries in two equally-sized tuples
template <typename... S, typename T, int... I>
constexpr auto mult_helper(const std::tuple<S...>& A, T scale, std::integer_sequence<int, I...>)
{
  return std::make_tuple(std::get<I>(A) * scale...);
}

// apply (scale *) to each entry in a tuple
template <typename S, typename... T, int... I>
constexpr auto mult_helper(S scale, const std::tuple<T...>& A, std::integer_sequence<int, I...>)
{
  return std::make_tuple((scale * std::get<I>(A))...);
}

// promote a double-precision value to a dual number representation that keeps track of
// derivatives w.r.t. more than 1 argument. It is assumed that 'arg' itself corresponds
// to the 'j'th argument. This function is not intended to be called outside of make_dual.
template <int j, int... i>
constexpr auto make_dual_helper(double arg, std::integer_sequence<int, i...>)
{
  using gradient_type = std::tuple<typename std::conditional<i == j, double, zero>::type...>;
  dual<gradient_type> arg_dual{};
  arg_dual.value                 = arg;
  std::get<j>(arg_dual.gradient) = 1.0;
  return arg_dual;
}

// promote a tensor of values to a dual tensor representation that keeps track of
// derivatives w.r.t. more than 1 argument. It is assumed that 'arg' itself corresponds
// to the 'j'th argument. This function is not intended to be called outside of make_dual.
template <int I, typename T, int... n, int... i>
constexpr auto make_dual_helper(const tensor<T, n...>& arg, std::integer_sequence<int, i...>)
{
  using gradient_type = std::tuple<typename std::conditional<i == I, tensor<T, n...>, zero>::type...>;
  tensor<dual<gradient_type>, n...> arg_dual{};
  for_constexpr<n...>([&](auto... j) {
    arg_dual(j...).value                       = arg(j...);
    std::get<I>(arg_dual(j...).gradient)(j...) = 1.0;
  });
  return arg_dual;
}

// promote a tuple of values to a tuple of dual value representations that keeps track of
// derivatives w.r.t. each tuple entry.
template <typename... T, int... i>
constexpr auto make_dual_helper(std::tuple<T...> args, std::integer_sequence<int, i...> seq)
{
  return std::make_tuple((make_dual_helper<i>(std::get<i>(args), seq))...);
}

// chain rule between a tuple of values and a single value,
// effectively equivalent to
//
// std::tuple df{
//   chain_rule(std::get<0>(df_dx), dx),
//   chain_rule(std::get<1>(df_dx), dx),
//   chain_rule(std::get<2>(df_dx), dx),
//   ...
// }
template <typename... T, typename S>
auto chain_rule_tuple_scale(std::tuple<T...> df_dx, S dx)
{
  return std::apply(
      [&](auto... each_component_of_df_dx) { return std::tuple{chain_rule(each_component_of_df_dx, dx)...}; }, df_dx);
}

template <typename... T, typename... S, int... i>
auto chain_rule_tuple_vecvec(std::tuple<T...> df_dx, std::tuple<S...> dx, std::integer_sequence<int, i...>)
{
  return (chain_rule(std::get<i>(df_dx), std::get<i>(dx)) + ...);
}

template <typename... T, typename... S>
auto chain_rule_tuple_matvec(std::tuple<T...> df_dx, std::tuple<S...> dx)
{
  auto int_seq = std::make_integer_sequence<int, int(sizeof...(S))>();

  return std::apply(
      [&](auto... each_component_of_df_dx) {
        return std::tuple{chain_rule_tuple_vecvec(each_component_of_df_dx, dx, int_seq)...};
      },
      df_dx);
}

}  // namespace detail

template <typename... S, typename... T>
constexpr auto operator+(const std::tuple<S...>& A, const std::tuple<T...>& B)
{
  static_assert(sizeof...(S) == sizeof...(T), "Error in operator+: tuple sizes must match");
  return detail::plus_helper(A, B, std::make_integer_sequence<int, int(sizeof...(S))>{});
}

template <typename... S, typename... T>
constexpr auto operator-(const std::tuple<S...>& A, const std::tuple<T...>& B)
{
  static_assert(sizeof...(S) == sizeof...(T), "Error in operator+: tuple sizes must match");
  return detail::minus_helper(A, B, std::make_integer_sequence<int, int(sizeof...(S))>{});
}

template <typename... S, typename T>
constexpr auto operator*(const std::tuple<S...>& A, T scale)
{
  return detail::mult_helper(A, scale, std::make_integer_sequence<int, int(sizeof...(S))>{});
}

template <typename S, typename... T>
constexpr auto operator*(S scale, const std::tuple<T...>& A)
{
  return detail::mult_helper(scale, A, std::make_integer_sequence<int, int(sizeof...(T))>{});
}

/**
 * @brief Constructs a tuple of dual numbers from a parameter pack of values
 * @param[in] args The set of values
 * The gradients for each value will be set to 1 (or its tensor equivalent)
 */
template <typename... T>
constexpr auto make_dual(T... args)
{
  return detail::make_dual_helper(std::tuple{args...}, std::make_integer_sequence<int, int(sizeof...(T))>{});
}
/// @overload
template <typename... T>
constexpr auto make_dual(std::tuple<T...> args)
{
  return detail::make_dual_helper(args, std::make_integer_sequence<int, int(sizeof...(T))>{});
}

/**
 * @brief Retrieves the value components of a set of (possibly dual) numbers
 * @param[in] tuple_of_values The tuple of numbers to retrieve values from
 * @pre The tuple must contain only scalars or tensors of @p dual numbers or doubles
 */
template <typename... T>
auto get_value(std::tuple<T...> tuple_of_values)
{
  return std::apply([](auto... each_value) { return std::tuple{get_value(each_value)...}; }, tuple_of_values);
}

/**
 * @brief Retrieves the gradient components of a set of dual numbers
 * @param[in] arg The set of numbers to retrieve gradients from
 */
template <typename... T>
auto get_gradient(dual<std::tuple<T...> > arg)
{
  return std::apply([](auto... each_value) { return std::tuple{each_value...}; }, arg.gradient);
}
/// @overload
template <typename... T, int... n>
auto get_gradient(const tensor<dual<std::tuple<T...> >, n...>& arg)
{
  std::tuple<outer_product_t<tensor<double, n...>, T>...> g{};
  for_constexpr<n...>([&](auto... i) {
    for_constexpr<sizeof...(T)>([&](auto j) { std::get<j>(g)(i...) = std::get<j>(arg(i...).gradient); });
  });
  return g;
}
/// @overload
template <typename... T>
auto get_gradient(std::tuple<T...> tuple_of_values)
{
  return std::apply([](auto... each_value) { return std::tuple{get_gradient(each_value)...}; }, tuple_of_values);
}

template <typename S, typename T>
auto chain_rule(S df_dx, T dx)
{
  if constexpr ((detail::is_tuple_of_tuples<S>::value && detail::is_tuple<T>::value)) {
    return detail::chain_rule_tuple_matvec(df_dx, dx);
  } else if constexpr (detail::is_tuple<S>::value && detail::is_tuple<T>::value) {
    auto int_seq = std::make_integer_sequence<int, int(std::tuple_size<T>{})>();
    return detail::chain_rule_tuple_vecvec(df_dx, dx, int_seq);
  } else if constexpr (detail::is_tuple<S>::value && !detail::is_tuple<T>::value) {
    return detail::chain_rule_tuple_scale(df_dx, dx);
  } else {
    return chain_rule(df_dx, dx);
  }
}

}  // namespace serac

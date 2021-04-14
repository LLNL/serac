#pragma once

#include <tuple>

#include "tensor.hpp"

template <typename T>
struct is_tuple {
  static constexpr bool value = false;
};

template <typename... T>
struct is_tuple<std::tuple<T...> > {
  static constexpr bool value = true;
};

template <typename T>
struct is_tuple_of_tuples {
  static constexpr bool value = false;
};

template <typename... T>
struct is_tuple_of_tuples<std::tuple<T...> > {
  static constexpr bool value = (is_tuple<T>::value && ...);
};

/////////////////////////////////////////////////

template <typename... S, typename... T, int... I>
constexpr auto plus_impl(const std::tuple<S...>& A, const std::tuple<T...>& B, std::integer_sequence<int, I...>)
{
  return std::make_tuple((std::get<I>(A) + std::get<I>(B))...);
}

template <typename... S, typename... T>
constexpr auto operator+(const std::tuple<S...>& A, const std::tuple<T...>& B)
{
  static_assert(sizeof...(S) == sizeof...(T), "Error in operator+: tuple sizes must match");
  return plus_impl(A, B, std::make_integer_sequence<int, int(sizeof...(S))>{});
}

template <typename... S, typename... T, int... I>
constexpr auto minus_impl(const std::tuple<S...>& A, const std::tuple<T...>& B, std::integer_sequence<int, I...>)
{
  return std::make_tuple((std::get<I>(A) - std::get<I>(B))...);
}

template <typename... S, typename... T>
constexpr auto operator-(const std::tuple<S...>& A, const std::tuple<T...>& B)
{
  static_assert(sizeof...(S) == sizeof...(T), "Error in operator+: tuple sizes must match");
  return minus_impl(A, B, std::make_integer_sequence<int, int(sizeof...(S))>{});
}

template <typename... S, typename T, int... I>
constexpr auto mult_impl(const std::tuple<S...>& A, T scale, std::integer_sequence<int, I...>)
{
  return std::make_tuple(std::get<I>(A) * scale...);
}

template <typename... S, typename T>
constexpr auto operator*(const std::tuple<S...>& A, T scale)
{
  return mult_impl(A, scale, std::make_integer_sequence<int, int(sizeof...(S))>{});
}

template <typename S, typename... T, int... I>
constexpr auto mult_impl(S scale, const std::tuple<T...>& A, std::integer_sequence<int, I...>)
{
  return std::make_tuple((scale * std::get<I>(A))...);
}

template <typename S, typename... T>
constexpr auto operator*(S scale, const std::tuple<T...>& A)
{
  return mult_impl(scale, A, std::make_integer_sequence<int, int(sizeof...(T))>{});
}

template <int I, int... i>
constexpr auto make_dual_helper(double arg, std::integer_sequence<int, i...>)
{
  using gradient_type = std::tuple<typename std::conditional<i == I, double, zero>::type...>;
  dual<gradient_type> arg_dual{};
  arg_dual.value                 = arg;
  std::get<I>(arg_dual.gradient) = 1.0;
  return arg_dual;
}

template <int I, typename T, int... n, int... i>
constexpr auto make_dual_helper(tensor<T, n...> arg, std::integer_sequence<int, i...>)
{
  using gradient_type = std::tuple<typename std::conditional<i == I, tensor<T, n...>, zero>::type...>;
  tensor<dual<gradient_type>, n...> arg_dual{};
  for_constexpr<n...>([&](auto... j) {
    arg_dual(j...).value                       = arg(j...);
    std::get<I>(arg_dual(j...).gradient)(j...) = 1.0;
  });
  return arg_dual;
}

template <typename... T, int... i>
constexpr auto make_dual_helper(std::tuple<T...> args, std::integer_sequence<int, i...> seq)
{
  return std::make_tuple((make_dual_helper<i>(std::get<i>(args), seq))...);
}

template <typename... T>
constexpr auto make_dual(T... args)
{
  return make_dual_helper(std::tuple{args...}, std::make_integer_sequence<int, int(sizeof...(T))>{});
}

template <typename... T>
constexpr auto make_dual(std::tuple<T...> args)
{
  return make_dual_helper(args, std::make_integer_sequence<int, int(sizeof...(T))>{});
}

template <typename... T>
auto get_value(std::tuple<T...> tuple_of_values)
{
  return std::apply([](auto... each_value) { return std::tuple{get_value(each_value)...}; }, tuple_of_values);
}

template <typename... T>
auto get_gradient(dual<std::tuple<T...> > arg)
{
  return std::apply([](auto... each_value) { return std::tuple{each_value...}; }, arg.gradient);
}

template <typename... T, int... n>
auto get_gradient(tensor<dual<std::tuple<T...> >, n...> arg)
{
  std::tuple<outer_product_t<tensor<double, n...>, T>...> g{};
  for_constexpr<n...>([&](auto... i) {
    for_constexpr<sizeof...(T)>([&](auto j) { std::get<j>(g)(i...) = std::get<j>(arg(i...).gradient); });
  });
  return g;
}

template <typename... T>
auto get_gradient(std::tuple<T...> tuple_of_values)
{
  return std::apply([](auto... each_value) { return std::tuple{get_gradient(each_value)...}; }, tuple_of_values);
}

template <typename... T, typename S, int... i>
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

template <typename S, typename T>
auto chain_rule(S df_dx, T dx)
{
  if constexpr ((is_tuple_of_tuples<S>::value && is_tuple<T>::value)) {
    return chain_rule_tuple_matvec(df_dx, dx);
  } else if constexpr (is_tuple<S>::value && is_tuple<T>::value) {
    auto int_seq = std::make_integer_sequence<int, int(std::tuple_size<T>{})>();
    return chain_rule_tuple_vecvec(df_dx, dx, int_seq);
  } else if constexpr (is_tuple<S>::value && !is_tuple<T>::value) {
    return chain_rule_tuple_scale(df_dx, dx);
  } else {
    return chain_rule(df_dx, dx);
  }
}
#pragma once

template <typename T>
struct FunctionSignature;

template <typename output_type, typename... input_types>
struct FunctionSignature<output_type(input_types...)> {
  using return_type     = output_type;
  using parameter_types = std::tuple<input_types...>;

  static constexpr int  num_args  = sizeof...(input_types);
  static constexpr auto index_seq = std::make_integer_sequence<int, num_args>{};
};

#include <camp/camp.hpp>
#pragma once

template <typename T>
struct FunctionSignature;

/**
 * @brief a type that encodes information about a function signature (return type, input types)
 * @tparam output_type the function signature's return type
 * @tparam ...input_types the function signature's input types
 */
template <typename output_type, typename... input_types>
struct FunctionSignature<output_type(input_types...)> {
  using return_type     = output_type;                 ///< the type returned by the function
  using parameter_types = std::tuple<input_types...>;  ///< the types of the function arguments

  /// the number of input arguments in the function signature
  static constexpr int num_args = sizeof...(input_types);

  /// integer sequence used to make iterating over arguments easier
  static constexpr auto index_seq = camp::make_int_seq<int, num_args>{};
};

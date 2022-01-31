// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "serac/infrastructure/accelerator.hpp"
#include "serac/numerics/functional/tuple.hpp"
#include "serac/numerics/functional/finite_element.hpp"

namespace serac {

/**
 * @brief a helper function for attaching the appropriate "shape" to the 1D containers
 *   mfem uses to pass data around. This way, the type itself encodes the dimensions so that
 *   the developer doens't have to reinterpret cast at the point of use
 *
 * @tparam exec for specifying whether or not the underlying data is on the CPU or GPU
 * @tparam element_type the finite element types whose data is stored in this container
 */
template <serac::ExecutionSpace exec, typename element_type>
auto ArrayViewForElement(const double* ptr, size_t num_elements, element_type)
{
  if constexpr (element_type::components == 1) {
    return ExecArrayView<const double, 2, exec>(ptr, num_elements, element_type::ndof);
  } else {
    return ExecArrayView<const double, 3, exec>(ptr, num_elements, element_type::components, element_type::ndof);
  }
}

/**
 * @brief a class for accessing E-vectors used by finite element kernels
 *
 * @tparam exec for specifying whether or not the underlying data is on the CPU or GPU
 * @tparam element_types the finite element types whose data is stored in this container
 */
template <serac::ExecutionSpace exec, typename... element_types>
struct EVectorView {
  static constexpr int n = sizeof...(element_types);  ///< how many element types were provided

  using element_types_tuple = serac::tuple<element_types...>;  ///< a list of the provided element types

  // clang-format off
  /// @brief the return type of the `operator[]` for this class
  using T = serac::tuple<
    typename std::conditional<
      element_types::components == 1, 
      tensor<double, element_types::ndof>,
      tensor<double, element_types::components, element_types::ndof> 
    >::type...
  >;
  // clang-format on

  /**
   * @brief Constructor for wrapping the raw pointers from mfem into multidimensional array views with the appropriate
   * dimensions
   *
   * @param pointers the list of raw pointers to E-vector data provided by mfem
   * @param num_elements the number of elements
   */
  EVectorView(std::array<const double*, n> pointers, size_t num_elements)
  {
    for_constexpr<n>([&](auto i) {
      serac::get<i>(data) = ArrayViewForElement<exec>(pointers[i], num_elements, serac::get<i>(element_types_tuple{}));
    });
  }

  /**
   * @brief for returning the values associated with a given element
   * @param e which element to retrieve data for
   */
  T operator[](size_t e)
  {
    T values{};

    for_constexpr<n>([&](auto I) {
      using element_type =
          typename serac::tuple_element<I,
                                        element_types_tuple>::type;  // decltype(serac::get<I>(element_types_tuple{}));
      constexpr int ndof       = element_type::ndof;
      constexpr int components = element_type::components;

      auto& arr = serac::get<I>(data);

      if constexpr (components == 1) {
        serac::get<I>(values) = make_tensor<ndof>([&arr, e](int i) { return arr(e, size_t(i)); });
      } else {
        serac::get<I>(values) =
            make_tensor<components, ndof>([&arr, e](int j, int i) { return arr(e, size_t(j), size_t(i)); });
      }
    });

    return values;
  }

  /**
   * @brief make an ArrayView for each of the element types, and make its dimension
   * 2 when spaces == 1  (num_elements, dofs_per_element)
   * 3 when spaces  > 1  (num_elements, dofs_per_element, num_components)
   */
  serac::tuple<serac::ExecArrayView<const double, 2 + (element_types::components > 1), exec>...> data;
};

}  // namespace serac

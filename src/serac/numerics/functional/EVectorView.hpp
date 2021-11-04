#pragma once

#include "serac/numerics/functional/array.hpp"
#include "serac/numerics/functional/tuple.hpp"
#include "serac/numerics/functional/finite_element.hpp"

namespace serac {

template < serac::ExecutionSpace exec, typename element_type >
auto ArrayViewForElement(double * ptr, size_t num_elements, element_type) {
  if constexpr (element_type::components == 1) {
    return ArrayView<double, 2, exec>(ptr, num_elements, element_type::ndof);
  } else {
    return ArrayView<double, 3, exec>(ptr, num_elements, element_type::ndof, element_type::components);
  }
}

template < serac::ExecutionSpace exec, typename ... element_types >
struct EVectorView {

  using element_types_tuple = serac::tuple< element_types ... >;

  static constexpr int n = sizeof ... (element_types);

  using T = serac::tuple < typename element_types::residual_type ... >;

  EVectorView(std::array<double *, n> pointers, size_t num_elements) {
    for_constexpr< n >([&](auto i){
      serac::get<i>(data) = ArrayViewForElement<exec>(pointers[i], num_elements, serac::get<i>(element_types_tuple{}));
    });
  }

  void update_pointers(std::array<double *, n> pointers) {
    for_constexpr< n >([&](auto i){
      serac::get<i>(data).ptr = pointers[i];
    });
  }

  T operator[](size_t e) {

    T values{};

    for_constexpr< n >([&](auto I){
      using element_type = decltype(serac::get<I>(element_types_tuple{}));
      constexpr int ndof = element_type::ndof;
      constexpr int components = element_type::components;

      auto & arr = serac::get<I>(data);

      if constexpr (components == 1) {
        serac::get<I>(values) = make_tensor<ndof>([&arr, e](int i) { return arr(size_t(i), e); });
      } else {
        serac::get<I>(values) = make_tensor<ndof, components>([&arr, e](int i, int j) { return arr(size_t(i), size_t(j), e); });
      }
    });

    return values;
 
  }

  /** 
   * @brief make an ArrayView for each of the element types, and make its dimension
   * 2 when spaces == 1  (num_elements, dofs_per_element)
   * 3 when spaces  > 1  (num_elements, dofs_per_element, num_components)
   */
  serac::tuple < serac::ArrayView< double, 2 + (element_types::components > 1), exec > ... > data;

};

}

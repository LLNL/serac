// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file functional_material_utils.hpp
 *
 * @brief Utilities for compile-time checking of types for user-defined loads and materials
 */

#pragma once

#include "serac/numerics/functional/functional.hpp"

namespace serac {

// Use SFINAE to add static assertions checking if the given thermal material type is acceptable
template <typename T, int dim, typename = void>
struct has_density : std::false_type {
};

template <typename T, int dim>
struct has_density<T, dim, std::void_t<decltype(std::declval<T&>().density(std::declval<tensor<double, dim>&>()))>>
    : std::true_type {
};

template <typename T, typename = void>
struct is_parameterized : std::false_type {
};

template <typename T>
struct is_parameterized<T, std::void_t<decltype(std::declval<T&>().numParameters())>> : std::true_type {
};

template <typename unparameterized_material_type>
struct trivially_parameterized_material {
  template <typename T1, typename T2, typename T3, typename... S>
  SERAC_HOST_DEVICE auto operator()(const T1& x, const T2& u, const T3& du_dx, S...) const
  {
    return mat(x, u, du_dx);
  }

  unparameterized_material_type mat;
};

template <typename T>
trivially_parameterized_material(T) -> trivially_parameterized_material<T>;

template <typename T>
auto parameterize_material(T& material)
{
  if constexpr (is_parameterized<T>::value) {
    return material;
  } else {
    return trivially_parameterized_material{material};
  }
}

template <typename unparameterized_source_type>
struct trivially_parameterized_source {
  template <typename T1, typename T2, typename T3, typename... S>
  SERAC_HOST_DEVICE auto operator()(const T1& x, double t, const T2& u, const T3& du_dx, S...) const
  {
    return source(x, t, u, du_dx);
  }

  unparameterized_source_type source;
};

template <typename T>
trivially_parameterized_source(T) -> trivially_parameterized_source<T>;

template <typename T>
auto parameterize_source(T& source)
{
  if constexpr (is_parameterized<T>::value) {
    return source;
  } else {
    return trivially_parameterized_source{source};
  }
}

template <typename unparameterized_flux_type>
struct trivially_parameterized_flux {
  template <typename T1, typename T2, typename T3, typename... S>
  SERAC_HOST_DEVICE auto operator()(const T1& x, const T2& n, const T3& u, S...) const
  {
    return flux(x, n, u);
  }

  unparameterized_flux_type flux;
};

template <typename T>
trivially_parameterized_flux(T) -> trivially_parameterized_flux<T>;

template <typename T>
auto parameterize_flux(T& flux)
{
  if constexpr (is_parameterized<T>::value) {
    return flux;
  } else {
    return trivially_parameterized_flux{flux};
  }
}

template <typename T, int dim, typename = void>
struct has_stress : std::false_type {
};

template <typename T, int dim>
struct has_stress<T, dim, std::void_t<decltype(std::declval<T&>()(std::declval<tensor<double, dim, dim>&>()))>>
    : std::true_type {
};

// Use SFINAE to add static assertions checking if the given thermal source type is acceptable
template <typename T, int dim, typename = void>
struct has_body_force : std::false_type {
};

template <typename T, int dim>
struct has_body_force<T, dim,
                      std::void_t<decltype(std::declval<T&>()(
                          std::declval<tensor<double, dim>&>(), std::declval<double>(),
                          std::declval<tensor<double, dim>&>(), std::declval<tensor<double, dim, dim>&>()))>>
    : std::true_type {
};

// Use SFINAE to add static assertions checking if the given thermal flux boundary type is acceptable
template <typename T, int dim, typename = void>
struct has_traction_boundary : std::false_type {
};

template <typename T, int dim>
struct has_traction_boundary<
    T, dim,
    std::void_t<decltype(std::declval<T&>()(std::declval<tensor<double, dim>&>(), std::declval<tensor<double, dim>&>(),
                                            std::declval<tensor<double, 1>&>()))>> : std::true_type {
};

// Use SFINAE to add static assertions checking if the given thermal flux boundary type is acceptable
template <typename T, int dim, typename = void>
struct has_pressure_boundary : std::false_type {
};

template <typename T, int dim>
struct has_pressure_boundary<
    T, dim,
    std::void_t<decltype(std::declval<T&>()(std::declval<tensor<double, dim>&>(), std::declval<tensor<double, 1>&>()))>>
    : std::true_type {
};

}  // namespace serac

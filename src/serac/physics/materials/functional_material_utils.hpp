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

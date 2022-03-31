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

template <typename T, typename = void>
struct is_parameterized : std::false_type {
};

template <typename T>
struct is_parameterized<T, std::void_t<decltype(std::declval<T&>().numParameters())>> : std::true_type {
};

/**
 * @brief Wrapper to treat an unparameterized material as a parameterized one
 *
 * @tparam UnparameterizedMaterialType The unparameterized material
 */
template <typename UnparameterizedMaterialType>
struct TriviallyParameterizedMaterial {
  /**
   * @brief Material response wrapper for an unparameterized material in a parameterized context
   *
   * @tparam T1 Spatial position type
   * @tparam T2 Temperature type
   * @tparam T3 Temperature gradient type
   * @tparam S Unused parameter pack type
   * @param x Spatial position
   * @param u Temperature
   * @param du_dx Temperature gradient
   * @return Material response of the unparameterized material
   */
  template <typename T1, typename T2, typename T3, typename... S>
  SERAC_HOST_DEVICE auto operator()(const T1& x, const T2& u, const T3& du_dx, S...) const
  {
    return mat(x, u, du_dx);
  }

  /// Underlying unparameterized material
  UnparameterizedMaterialType mat;
};

/**
 * @brief Template deduction guide for the trivially parameterized material
 *
 * @tparam T The unparameterized material type
 */
template <typename T>
TriviallyParameterizedMaterial(T) -> TriviallyParameterizedMaterial<T>;

/**
 * @brief Convert an unparameterized material to one which accepts parameter values in the paren operator
 *
 * @tparam T The unparameterized material type
 * @param material The unparameterized material
 * @return The parameterized material
 */
template <typename T>
auto parameterizeMaterial(T& material)
{
  if constexpr (is_parameterized<T>::value) {
    return material;
  } else {
    return TriviallyParameterizedMaterial{material};
  }
}

/**
 * @brief Wrapper to treat an unparameterized source as a parameterized one
 *
 * @tparam UnparameterizedSourceType The unparameterized source
 */
template <typename UnparameterizedSourceType>
struct TriviallyParameterizedSource {
  /**
   * @brief Wrapper for an unparameterized source in a parameterized context
   *
   * @tparam T1 Spatial position type
   * @tparam T2 Temperature type
   * @tparam T3 Temperature gradient type
   * @tparam S Unused parameter pack type
   * @param x Spatial position
   * @param t Time
   * @param u Temperature
   * @param du_dx Temperature gradient
   * @return Volumetric source for the unparameterized source
   */
  template <typename T1, typename T2, typename T3, typename... S>
  SERAC_HOST_DEVICE auto operator()(const T1& x, double t, const T2& u, const T3& du_dx, S...) const
  {
    return source(x, t, u, du_dx);
  }

  /// Underlying unparameterized source
  UnparameterizedSourceType source;
};

/**
 * @brief Template deduction guide for the trivially parameterized source
 *
 * @tparam T The unparameterized source type
 */
template <typename T>
TriviallyParameterizedSource(T) -> TriviallyParameterizedSource<T>;

/**
 * @brief Convert an unparameterized source to one which accepts parameter values in the paren operator
 *
 * @tparam T The unparameterized source type
 * @param source The unparameterized source
 * @return The parameterized source
 */
template <typename T>
auto parameterizeSource(T& source)
{
  if constexpr (is_parameterized<T>::value) {
    return source;
  } else {
    return TriviallyParameterizedSource{source};
  }
}

/**
 * @brief Wrapper for unparameterized boundary flux types to be used in a parameterized setting
 *
 * @tparam UnparameterizedFluxType The unparameterized boundary flux type
 */
template <typename UnparameterizedFluxType>
struct TriviallyParameterizedFlux {
  /**
   * @brief The wrapper for an unparameterized boundary flux object to be called using the parameterized
   * call signature
   *
   * @tparam T1 Spatial position type
   * @tparam T2 Normal vector type
   * @tparam T3 Temperature type
   * @tparam S Unused parameter pack type
   * @param x Spatial position
   * @param n Normal vector
   * @param u Temperature
   * @return Computed boundary flux to be applied
   */
  template <typename T1, typename T2, typename T3, typename... S>
  SERAC_HOST_DEVICE auto operator()(const T1& x, const T2& n, const T3& u, S...) const
  {
    return flux(x, n, u);
  }

  /// Underlying unparameterized flux
  UnparameterizedFluxType flux;
};

/**
 * @brief Template deduction guide for the trivially parameterized flux
 *
 * @tparam T The unparameterized flux type
 */
template <typename T>
TriviallyParameterizedFlux(T) -> TriviallyParameterizedFlux<T>;

/**
 * @brief Convert an unparameterized flux to one which accepts parameter values in the paren operator
 *
 * @tparam T The unparameterized flux type
 * @param flux The unparameterized flux
 * @return The parameterized flux
 */
template <typename T>
auto parameterizeFlux(T& flux)
{
  if constexpr (is_parameterized<T>::value) {
    return flux;
  } else {
    return TriviallyParameterizedFlux{flux};
  }
}

/**
 * @brief Wrapper to treat an unparameterized source as a parameterized one
 *
 * @tparam UnparameterizedSourceType The unparameterized source
 */
template <typename UnparameterizedPressureType>
struct TriviallyParameterizedPressure {
  /**
   * @brief Wrapper for an unparameterized source in a parameterized context
   *
   * @tparam T1 Spatial position type
   * @tparam T2 Temperature type
   * @tparam T3 Temperature gradient type
   * @tparam S Unused parameter pack type
   * @param x Spatial position
   * @param t Time
   * @return Volumetric source for the unparameterized source
   */
  template <typename T1, typename... S>
  SERAC_HOST_DEVICE auto operator()(const T1& x, double t, S...) const
  {
    return pressure(x, t);
  }

  /// Underlying unparameterized source
  UnparameterizedPressureType pressure;
};

/**
 * @brief Template deduction guide for the trivially parameterized source
 *
 * @tparam T The unparameterized source type
 */
template <typename T>
TriviallyParameterizedPressure(T) -> TriviallyParameterizedPressure<T>;

/**
 * @brief Convert an unparameterized pressure to one which accepts parameter values in the paren operator
 *
 * @tparam T The unparameterized pressure type
 * @param pressure The unparameterized pressure
 * @return The parameterized pressure
 */
template <typename T>
auto parameterizePressure(T& pressure)
{
  if constexpr (is_parameterized<T>::value) {
    return pressure;
  } else {
    return TriviallyParameterizedPressure{pressure};
  }
}

}  // namespace serac

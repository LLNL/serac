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
   * @tparam PositionType Spatial position type
   * @tparam StateType State type
   * @tparam StateGradType State gradient type
   * @tparam ParamTypes Unused parameter pack type
   * @param x Spatial position
   * @param u State
   * @param du_dx State gradient
   * @return Material response of the unparameterized material
   */
  template <typename PositionType, typename StateType, typename StateGradType, typename... ParamTypes>
  SERAC_HOST_DEVICE auto operator()(const PositionType& x, const StateType& u, const StateGradType& du_dx,
                                    ParamTypes...) const
  {
    return mat(x, u, du_dx);
  }

  /// Underlying unparameterized material
  UnparameterizedMaterialType mat;
};

/**
 * @brief Template deduction guide for the trivially parameterized material
 *
 * @tparam MaterialType The unparameterized material type
 */
template <typename MaterialType>
TriviallyParameterizedMaterial(MaterialType) -> TriviallyParameterizedMaterial<MaterialType>;

/**
 * @brief Convert an unparameterized material to one which accepts parameter values in the paren operator
 *
 * @tparam MaterialType The unparameterized material type
 * @param material The unparameterized material
 * @return The parameterized material
 */
template <typename MaterialType>
auto parameterizeMaterial(MaterialType& material)
{
  if constexpr (is_parameterized<MaterialType>::value) {
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
   * @tparam PositionType Spatial position type
   * @tparam StateType State type
   * @tparam StateGradType State gradient type
   * @tparam ParamTypes Unused parameter pack type
   * @param x Spatial position
   * @param t Time
   * @param u State
   * @param du_dx State gradient
   * @return Volumetric source for the unparameterized source
   */
  template <typename PositionType, typename StateType, typename StateGradType, typename... ParamTypes>
  SERAC_HOST_DEVICE auto operator()(const PositionType& x, double t, const StateType& u, const StateGradType& du_dx,
                                    ParamTypes...) const
  {
    return source(x, t, u, du_dx);
  }

  /// Underlying unparameterized source
  UnparameterizedSourceType source;
};

/**
 * @brief Template deduction guide for the trivially parameterized source
 *
 * @tparam SourceType The unparameterized source type
 */
template <typename SourceType>
TriviallyParameterizedSource(SourceType) -> TriviallyParameterizedSource<SourceType>;

/**
 * @brief Convert an unparameterized source to one which accepts parameter values in the paren operator
 *
 * @tparam SourceType The unparameterized source type
 * @param source The unparameterized source
 * @return The parameterized source
 */
template <typename SourceType>
auto parameterizeSource(SourceType& source)
{
  if constexpr (is_parameterized<SourceType>::value) {
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
   * @tparam PositionType Spatial position type
   * @tparam NormalType Normal vector type
   * @tparam StateType State type
   * @tparam ParamTypes Unused parameter pack type
   * @param x Spatial position
   * @param n Normal vector
   * @param u State
   * @return Computed boundary flux to be applied
   */
  template <typename PositionType, typename NormalType, typename StateType, typename... ParamTypes>
  SERAC_HOST_DEVICE auto operator()(const PositionType& x, const NormalType& n, const StateType& u, ParamTypes...) const
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
 * @tparam FluxType The unparameterized flux type
 * @param flux The unparameterized flux
 * @return The parameterized flux
 */
template <typename FluxType>
auto parameterizeFlux(FluxType& flux)
{
  if constexpr (is_parameterized<FluxType>::value) {
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
   * @tparam PositionType Spatial position type
   * @tparam ParamTypes Unused parameter pack type
   * @param x Spatial position
   * @param t Time
   * @return Volumetric source for the unparameterized source
   */
  template <typename PositionType, typename... ParamTypes>
  SERAC_HOST_DEVICE auto operator()(const PositionType& x, double t, ParamTypes...) const
  {
    return pressure(x, t);
  }

  /// Underlying unparameterized source
  UnparameterizedPressureType pressure;
};

/**
 * @brief Template deduction guide for the trivially parameterized source
 *
 * @tparam PressureType The unparameterized source type
 */
template <typename PressureType>
TriviallyParameterizedPressure(PressureType) -> TriviallyParameterizedPressure<PressureType>;

/**
 * @brief Convert an unparameterized pressure to one which accepts parameter values in the paren operator
 *
 * @tparam PressureType The unparameterized pressure type
 * @param pressure The unparameterized pressure
 * @return The parameterized pressure
 */
template <typename PressureType>
auto parameterizePressure(PressureType& pressure)
{
  if constexpr (is_parameterized<PressureType>::value) {
    return pressure;
  } else {
    return TriviallyParameterizedPressure{pressure};
  }
}

}  // namespace serac

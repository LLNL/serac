// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file quadrature.hpp
 *
 * @brief Definitions of quadrature rules for quads and hexes
 */

#pragma once

#include "tensor.hpp"
#include "polynomials.hpp"
#include "finite_element.hpp"

namespace serac {

/**
 * @brief A rule for numerical quadrature (set of points and weights)
 * Can be thought of as a compile-time analogue of @p mfem::IntegrationRule
 * @tparam n The number of points in the rule
 * @tparam dim The spatial dimension of integration
 */
template <int n, int dim>
struct QuadratureRule {
  /**
   * @brief The scalar weights of each point
   */
  tensor<double, n> weights;
  /**
   * @brief The coordinates in reference space for each quadrature point
   */
  reduced_tensor<double, n, dim> points;
  /**
   * @brief Returns the number of points in the rule
   */
  constexpr size_t size() const { return n; }
};

/**
 * @brief Returns the Gauss-Legendre quadrature rule for an element and order
 * @tparam The shape of the element to produce a quadrature rule for
 * @tparam Q the number of quadrature points per dimension
 */
template <Geometry g, int Q>
constexpr auto GaussQuadratureRule()
{
  auto x = GaussLegendreNodes<Q>();
  auto w = GaussLegendreWeights<Q>();

  if constexpr (g == Geometry::Segment) {
    return QuadratureRule<Q, 1>{w, x};
  }

  if constexpr (g == Geometry::Quadrilateral) {
    QuadratureRule<Q * Q, 2> rule{};
    int                      count = 0;
    for (int j = 0; j < Q; j++) {
      for (int i = 0; i < Q; i++) {
        rule.points[count]    = {x[i], x[j]};
        rule.weights[count++] = w[i] * w[j];
      }
    }
    return rule;
  }

  if constexpr (g == Geometry::Hexahedron) {
    QuadratureRule<Q * Q * Q, 3> rule{};
    int                          count = 0;
    for (int k = 0; k < Q; k++) {
      for (int j = 0; j < Q; j++) {
        for (int i = 0; i < Q; i++) {
          rule.points[count]    = {x[i], x[j], x[k]};
          rule.weights[count++] = w[i] * w[j] * w[k];
        }
      }
    }
    return rule;
  }
}

}  // namespace serac

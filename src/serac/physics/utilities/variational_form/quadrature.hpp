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
  array<double, n> weights;
  /**
   * @brief The coordinates in reference space for each quadrature point
   */
  array<tensor<double, dim>, n> points;
  /**
   * @brief Returns the number of points in the rule
   */
  constexpr size_t size() const { return n; }
};

/**
 * @brief Returns the quadrature rule for an element and order
 * @tparam The shape of the element to produce a quadrature rule for
 * @tparam Q The "order" of integration
 */
template <Geometry g, int Q>
constexpr auto GaussQuadratureRule()
{
  auto x = GaussLegendreNodes<Q>(0.0, 1.0);
  auto w = GaussLegendreWeights<Q>();
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

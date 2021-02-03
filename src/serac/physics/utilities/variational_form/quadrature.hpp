#pragma once

#include "tensor.hpp"
#include "polynomials.hpp"
#include "finite_element.hpp"

template <int n, int dim>
struct QuadratureRule {
  array<double, n>              weights;
  array<tensor<double, dim>, n> points;
  constexpr size_t              size() const { return n; }
};

template <::Geometry g, int Q>
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
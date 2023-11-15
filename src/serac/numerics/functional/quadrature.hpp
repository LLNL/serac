// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
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

struct PolynomialOrder { uint32_t p; };
struct PointsPerDimension { uint32_t q; };

struct QuadratureRule {
  std::vector< double > points;
  std::vector< double > weights;
  bool is_structured;
};

QuadratureRule GaussLegendreRule(mfem::Geometry::Type geom, PolynomialOrder order);
QuadratureRule GaussLegendreRule(mfem::Geometry::Type geom, PointsPerDimension n);

}  // namespace serac

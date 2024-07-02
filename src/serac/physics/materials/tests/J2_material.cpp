// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file J2_material.cpp
 *
 * @brief a verification problem for the J2 material model, taken from example 2 of
 * R. M. Brannon · S. Leelavanichkul (2009)
 * A multi-stage return algorithm for solving the classical
 * damage component of constitutive models for rocks,
 * ceramics, and other rock-like media
 *
 * @note Sam: the graphs for the analytic stress solutions in the document above
 * are inconsistent with the equations they provide for t > 1. It seems that
 * a couple of the coefficents in the equations appear with the wrong sign,
 * but the implementation below fixes those typos
 */

#include <iostream>

#include "serac/physics/materials/solid_material.hpp"
#include "serac/physics/materials/material_verification_tools.hpp"

using namespace serac;

tensor<double, 3, 3> analytic_soln(double t)
{
  // clang-format off
  double a = exp(12.33 * t);

  tensor< double, 3 > sigma{};

  if (t <= 0.201) {
    sigma = {-474.0 * t, -474.0 * t, 948.0 * t};
  }

  if ((0.201 < t) && (t <= 1.0)) {
    sigma = {-95.26, -95.26, 190.50};
  }

  if (1 < t) {
    tensor < double, 3, 4 > c = {{
      { 189.40,  0.1704, -0.003242, 0.00001712},
      { -76.87, -1.4430,  0.001316, 0.00001712},
      {-112.50,  1.2720,  0.001926, 0.00001712}
    }};

    sigma = {
      (c(0, 0) + c(0, 1) * sqrt(a) + c(0, 2) * a) / (1.0 + c(0, 3) * a),
      (c(1, 0) + c(1, 1) * sqrt(a) + c(1, 2) * a) / (1.0 + c(1, 3) * a),
      (c(2, 0) + c(2, 1) * sqrt(a) + c(2, 2) * a) / (1.0 + c(2, 3) * a)
    };
  }

  return {{
    {sigma[0], 0.0, 0.0}, 
    {0.0, sigma[1], 0.0}, 
    {0.0, 0.0, sigma[2]}
  }};
  // clang-format on
}

int main()
{
  double tmax      = 2.0;
  size_t num_steps = 64;

  double G = 79000;
  double K = 10 * G;

  double E  = 9 * K * G / (3 * K + G);
  double nu = (3 * K - 2 * G) / (2 * (3 * K + G));
  double sigma_y = 165 * sqrt(3.0);

  using Hardening = solid_mechanics::LinearHardening;
  using Material = solid_mechanics::J2Nonlinear<Hardening>;

  Hardening hardening{.sigma_y = sigma_y, .Hi = 0.0};
  Material material{.E = E, .nu = nu, .hardening = hardening, .Hk = 0.0, .density = 1.0};
  Material::State initial_state{};

  tensor<double, 3> epsilon[2] = {{-0.0030000, -0.003, 0.0060000}, {-0.0103923, 0.000, 0.0103923}};

  auto du_dX = [=](double t) {
    double a[2] = {(t < 1) ? t : 2.0 - t, (t < 1) ? 0 : t - 1.0};
    return diag(a[0] * epsilon[0] + a[1] * epsilon[1]);
  };

  auto history = single_quadrature_point_test(tmax, num_steps, material, initial_state, du_dX);

  for (auto [t, state, dudx, stress] : history) {
    if (t > 0) {
      double rel_error = norm(stress - analytic_soln(t)) / norm(stress);
      std::cout << t << ": " << rel_error << std::endl;
    }

    // for generating a plot like in the paper:
    // std::cout << "{" << t << ", " << stress[0][0] << ", " << stress[1][1] << ", " << stress[2][2] << "}" <<
    // std::endl;
  }
}

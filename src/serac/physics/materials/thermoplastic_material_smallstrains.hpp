// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermoplastic_material.hpp
 *
 *
 * @brief The material types for the thermoplastic functional physics module
 */

#pragma once

#include "serac/numerics/functional/functional.hpp"

/// ThermoMechanics helper data types
namespace serac {

/**
 * @brief Voce's isotropic hardening law
 *
 * This form has an exponential saturation character.
 */
struct VoceHardeningThermal {
  double sigma_y;          ///< yield strength
  double sigma_sat;        ///< saturation value of flow strength
  double strain_constant;  ///< The constant dictating how fast the exponential decays
  // double theta_ref;        /// Reference T (units K)
  // double omega;            /// Flow stress softening (units: K^-1)

  /**
   * @brief Computes the flow stress
   *
   * @tparam T Number-like type for the argument
   * @param accumulated_plastic_strain The uniaxial equivalent accumulated plastic strain
   * @return Flow stress value
   */
  template <typename T1>
  auto operator()(const T1 accumulated_plastic_strain) const
  {
    using std::exp;
    // return sigma_y + sigma_sat * exp(1 - accumulated_plastic_strain / strain_constant);
    //  * (1 - omega * (theta - theta_ref));
    return sigma_sat - (sigma_sat - sigma_y) * exp(-accumulated_plastic_strain / strain_constant);
  };
};

/// @brief J2 material with nonlinear isotropic hardening and thermal effects.
template <typename HardeningType>
struct J2NonlinearThermal {
  static constexpr int    dim = 3;      ///< spatial dimension
  static constexpr double tol = 1e-10;  ///< relative tolerance on residual to judge convergence of return map

  double        E;          ///< Young's modulus
  double        nu;         ///< Poisson's ratio
  HardeningType hardening;  ///< Flow stress hardening model
  double        density;    ///< mass density
  double        k;          ///< thermal conductivity
  double        C_v;        ///< volumetric heat capacity
  double        omega;      // thermal_softening
  double        theta_ref;  // reference temperature
  double        eta;        // efficiency

  /// @brief variables required to characterize the hysteresis response
  struct State {
    tensor<double, dim, dim> plastic_strain;              ///< plastic strain
    double                   accumulated_plastic_strain;  ///< uniaxial equivalent plastic strain
  };

  /** @brief */
  template <typename T1, typename T2, typename T3>
  auto operator()(State& state, const tensor<T1, 3, 3>& du_dX, const T2 theta, const tensor<T3, 3>& grad_theta) const
  {
    using std::sqrt;
    constexpr auto I = Identity<dim>();
    const double   K = E / (3.0 * (1.0 - 2.0 * nu));
    const double   G = 0.5 * E / (1.0 + nu);

    // (i) elastic predictor
    auto plastic_strain_old = state.plastic_strain;
    auto el_strain          = sym(du_dX) - plastic_strain_old;
    auto p                  = K * tr(el_strain);
    auto s                  = 2.0 * G * dev(el_strain);
    auto q                  = sqrt(1.5) * norm(s);

    // (ii) admissibility
    const double eqps_old         = state.accumulated_plastic_strain;
    const double softening_factor = 1 - omega * (get_value(theta) - theta_ref);
    auto         residual         = [eqps_old, G, softening_factor, *this](auto delta_eqps, auto trial_mises) {
      return trial_mises - 3.0 * G * delta_eqps - this->hardening(eqps_old + delta_eqps) * softening_factor;
    };

    // (iii) return mapping
    if (residual(0.0, get_value(q)) > tol * hardening.sigma_y) {
      ScalarSolverOptions opts{.xtol = 0, .rtol = tol * hardening.sigma_y, .max_iter = 25};
      double              lower_bound = 0.0;
      double              upper_bound = (get_value(q) - hardening(eqps_old)) / (3.0 * G);
      auto [delta_eqps, status]       = solve_scalar_equation(residual, 0.0, lower_bound, upper_bound, opts, q);

      //
      auto Np = 1.5 * s / q;

      //
      s = s - 2.0 * G * delta_eqps * Np;
      state.accumulated_plastic_strain += get_value(delta_eqps);
      state.plastic_strain += get_value(delta_eqps) * get_value(Np);
    }

    const auto sigma = s + p * I;

    // internal heat source
    const auto delta_plastic_strain = state.plastic_strain; //plastic_strain_old;  // needs /delta_t
    const auto s0 =
        double_dot(sigma, delta_plastic_strain) * eta;  // eta, efficiency of conversion plastic work -> heat

    // heat flux
    const auto q0 = -k * grad_theta;

    return serac::tuple{sigma, C_v, s0, q0}; // (Fabio's comment) as in green_saint_venant_themoelastic.hpp. 
    //However, it looks to me that the sorce term s0 is not used in the Thermal material interface
  };
};

}  // namespace serac

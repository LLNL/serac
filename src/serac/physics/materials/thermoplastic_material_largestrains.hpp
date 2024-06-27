// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermoplastic_material_large_strains.hpp
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
struct LargeStrainsVoceHardeningThermal {
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
struct J2LogStrainsThermal1 {
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

  /// @brief state variables
  struct State {
    tensor<double, dim, dim> F;                           ///< deformation gradient
    tensor<double, dim, dim> eps_el;                      ///< elastic strain
    double                   accumulated_plastic_strain;  ///< accumulated plastic strain
  };

  /** @brief */
  template <typename T1, typename T2, typename T3>
  auto operator()(State& state, const tensor<T1, 3, 3>& du_dX, const T2 theta, const tensor<T3, 3>& grad_theta) const
  {
    // Get states
    auto F_n = state.F;
    if (F_n[0][0] == 0) {
      F_n[0][0] = 1;
      F_n[1][1] = 1;
      F_n[2][2] = 1;
    }
    const auto eps_el_n             = state.eps_el;
    const double acc_plastic_strain = state.accumulated_plastic_strain;

    //
    constexpr auto I = Identity<dim>();
    const double   K = E / (3.0 * (1.0 - 2.0 * nu));
    const double   G = 0.5 * E / (1.0 + nu);

    // (i) elastic predictor
    // Compute current deformation gradient
    auto F = I + du_dX;

    // Retrieve be_n = exp(2 * e_n)
    auto be_n = exp_symm(2 * eps_el_n);

    //- Compute e_elastic = 0.5 ln(be)
    // delta_F = F * F_inv and delta_F^T = F_n_inv^T * F^T
    auto F_n_inv = inv(F_n);
    auto delta_F = dot(F, F_n_inv);

    // be = delta_F * bn * delta_F^T
    auto delta_F_T      = transpose(delta_F);
    auto be_n_delta_F_T = dot(be_n, delta_F_T);
    auto be             = dot(delta_F, be_n_delta_F_T);

    // e_elastic = 0.5 ln(be)
    auto eps_el_trial = 0.5 * log_symm(be);
    auto eps_el_dev_trial = dev(eps_el_trial);
    const auto eps_el_vol = eps_el_trial - eps_el_dev_trial;

    //
    auto s_trial = 2. * G * eps_el_dev_trial;
    auto q_trial = std::sqrt(1.5) * norm(s_trial);

    // (ii) admissibility
    const double softening_factor = 1 - omega * (get_value(theta) - theta_ref);
    auto         residual         = [acc_plastic_strain, G, softening_factor, *this](auto delta_gamma, auto trial_mises) {
      return trial_mises - 3.0 * G * delta_gamma - this->hardening(acc_plastic_strain + delta_gamma) * softening_factor;
    };

    double delta_gamma_value = 0.0;
    // (iii) return mapping
    if (residual(0.0, get_value(q_trial)) > tol * hardening.sigma_y) {
      ScalarSolverOptions opts{.xtol = 0, .rtol = tol * hardening.sigma_y, .max_iter = 25};
      double              lower_bound = 0.0;
      double              upper_bound = (get_value(q_trial) - hardening(acc_plastic_strain)) / (3.0 * G);
      auto [delta_gamma, status]       = solve_scalar_equation(residual, 0.0, lower_bound, upper_bound, opts, q_trial);

      //
      eps_el_dev_trial -= (3. * G * delta_gamma / q_trial) * eps_el_dev_trial;
      eps_el_trial -= (3. * G * delta_gamma / q_trial) * eps_el_dev_trial;
      delta_gamma_value = get_value(delta_gamma);
    }
    
    const auto s_updated = 2. * G * eps_el_dev_trial;
    
    // update states
    state.accumulated_plastic_strain += delta_gamma_value;
    state.F = get_value(F); 
    state.eps_el = get_value(eps_el_trial);

    const auto tau = s_updated + K * eps_el_vol;
    
    const auto sigma = tau / det(F);

    const auto q = std::sqrt(1.5) * norm(s_updated);

    // internal heat source
    //const auto delta_plastic_strain = state.plastic_strain;  // plastic_strain_old;  // needs /delta_t
    const auto s0 = q * delta_gamma_value * eta;
        //double_dot(sigma, delta_plastic_strain) * eta;  // eta, efficiency of conversion plastic work -> heat

    // heat flux
    const auto q0 = -k * grad_theta;

    return serac::tuple{sigma, C_v, s0, q0};  // (Fabio's comment) as in green_saint_venant_themoelastic.hpp.
    // However, it looks to me that the sorce term s0 is not used in the Thermal material interface
  };
};

/// @brief Second version based on Brandon's log_strain_plasticity implementation (of the same fomulation)
template <typename HardeningType>
struct J2LogStrainsThermal2 {
  static constexpr int    dim = 3;      ///< spatial dimension
  static constexpr double tol = 1e-10;  ///< relative tolerance on residual mag to judge convergence of return map

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
    tensor<double, dim, dim> Fpinv = DenseIdentity<3>(); ///< plastic distortion tensor
    double accumulated_plastic_strain;  ///< uniaxial equivalent plastic strain
  };

  /** @brief calculate the Cauchy stress, given the displacement gradient and previous material state */
  template <typename T, typename T2, typename T3>
  auto operator()(State& state, const T du_dX, const T2 theta, const tensor<T3, 3>& grad_theta) const
  {
    using std::sqrt;
    constexpr auto I = Identity<dim>();
    const double   K = E / (3.0 * (1.0 - 2.0 * nu));
    const double   G = 0.5 * E / (1.0 + nu);

    // (i) elastic predictor
    auto F = du_dX + I;
    auto Fe = dot(F, state.Fpinv);
    auto Ee = 0.5*log_symm(dot(transpose(Fe), Fe));
    auto p = K * tr(Ee);
    auto s = 2.0 * G * dev(Ee);
    auto q = sqrt(1.5) * norm(s);

    // (ii) admissibility
    const double eqps_old = state.accumulated_plastic_strain;
    const double softening_factor = 1 - omega * (get_value(theta) - theta_ref);
    auto         residual = [eqps_old, G, softening_factor, *this](auto delta_eqps, auto trial_mises) {
      return trial_mises - 3.0 * G * delta_eqps - this->hardening(eqps_old + delta_eqps) * softening_factor;
    };

    if (residual(0.0, get_value(q)) > tol * hardening.sigma_y) {
      // (iii) return mapping

      // Note the tolerance for convergence is the same as the tolerance for entering the return map.
      // This ensures that if the constitutive update is called again with the updated internal
      // variables, the return map won't be repeated.
      ScalarSolverOptions opts{.xtol = 0, .rtol = tol * hardening.sigma_y, .max_iter = 25};
      double              lower_bound = 0.0;
      double              upper_bound = (get_value(q) - hardening(eqps_old)) / (3.0 * G);
      auto [delta_eqps, status]       = solve_scalar_equation(residual, 0.0, lower_bound, upper_bound, opts, q);

      auto Np = 1.5 * s / q;

      s = s - 2.0 * G * delta_eqps * Np;
      auto A = exp_symm(-delta_eqps*Np);
      Fe = dot(Fe, A);
      state.accumulated_plastic_strain += get_value(delta_eqps);
      state.Fpinv = dot(state.Fpinv, get_value(A));
    }
    auto M = s + p * I; // Mandel stress

    auto sigma = dot(dot(Fe, M), inv(Fe))/det(F);

        // internal heat source
    const auto delta_plastic_strain = state.accumulated_plastic_strain -  eqps_old;  // plastic_strain_old;  // needs /delta_t
    const auto s0 = q * delta_plastic_strain * eta;
        //double_dot(sigma, delta_plastic_strain) * eta;  // eta, efficiency of conversion plastic work -> heat

    // heat flux
    const auto q0 = -k * grad_theta;

    return serac::tuple{sigma, C_v, s0, q0};  // (Fabio's comment) as in green_saint_venant_themoelastic.hpp.
    // However, it looks to me that the sorce term s0 is not used in the Thermal material interface

  }
};


}  // namespace serac

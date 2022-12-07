// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_material.hpp
 *
 * @brief The material and load types for the solid functional physics module
 */

#pragma once

#include "serac/numerics/functional/functional.hpp"

/// SolidMechanics helper data types
namespace serac::solid_mechanics {

/**
 * @brief Linear isotropic elasticity material model
 *
 */
struct LinearIsotropic {
  using State = Empty;  ///< this material has no internal variables

  /**
   * @brief stress calculation for a linear isotropic material model
   *
   * When applied to 2D displacement gradients, the stress is computed in plane strain,
   * returning only the in-plane components.
   *
   * @tparam T Number-like type for the displacement gradient components
   * @tparam dim Dimensionality of space
   * @param du_dX Displacement gradient with respect to the reference configuration
   * @return The Cauchy stress
   */
  template <typename T, int dim>
  SERAC_HOST_DEVICE auto operator()(State& /* state */, const tensor<T, dim, dim>& du_dX) const
  {
    auto I       = Identity<dim>();
    auto lambda  = K - (2.0 / 3.0) * G;
    auto epsilon = 0.5 * (transpose(du_dX) + du_dX);
    return lambda * tr(epsilon) * I + 2.0 * G * epsilon;
  }

  double density;  ///< mass density
  double K;        ///< bulk modulus
  double G;        ///< shear modulus
};

/**
 * @brief Neo-Hookean material model
 *
 */
struct NeoHookean {
  using State = Empty;  ///< this material has no internal variables

  /**
   * @brief stress calculation for a NeoHookean material model
   *
   * When applied to 2D displacement gradients, the stress is computed in plane strain,
   * returning only the in-plane components.
   *
   * @tparam T Number-like type for the displacement gradient components
   * @tparam dim Dimensionality of space
   * @param du_dX displacement gradient with respect to the reference configuration (displacement_grad)
   * @return The Cauchy stress
   */
  template <typename T, int dim>
  SERAC_HOST_DEVICE auto operator()(State& /* state */, const tensor<T, dim, dim>& du_dX) const
  {
    using std::log;
    constexpr auto I         = Identity<dim>();
    auto           lambda    = K - (2.0 / 3.0) * G;
    auto           B_minus_I = du_dX * transpose(du_dX) + transpose(du_dX) + du_dX;
    auto           J         = det(I + du_dX);
    return (lambda * log(J) * I + G * B_minus_I) / J;
  }

  double density;  ///< mass density
  double K;        ///< bulk modulus
  double G;        ///< shear modulus
};

/**
 * @brief Power-law isotropic hardening law
 */
struct PowerLawHardening {
  double sigma_y; ///< yield strength
  double n; ///< hardening index in reciprocal form
  double eps0; ///< reference value of accumulated plastic strain

  /**
   * @brief Computes the flow stress
   *
   * @tparam T Number-like type for the argument
   * @param accumulated_plastic_strain The uniaxial equivalent accumulated plastic strain
   * @return Flow stress value
   */
  template <typename T>
  auto operator()(const T accumulated_plastic_strain) const
  {
    using std::pow;
    return sigma_y*pow(1.0 + accumulated_plastic_strain/eps0, 1.0/n);
  };
};

/**
 * @brief Voce's isotropic hardening law
 *
 * This form has an exponential saturation character.
 */
struct VoceHardening {
  double sigma_y; ///< yield strength
  double sigma_sat; ///< saturation value of flow strength
  double strain_constant; ///< The constant dictating how fast the exponential decays

  /**
   * @brief Computes the flow stress
   *
   * @tparam T Number-like type for the argument
   * @param accumulated_plastic_strain The uniaxial equivalent accumulated plastic strain
   * @return Flow stress value
   */
  template <typename T>
  auto operator()(const T accumulated_plastic_strain) const
  {
    using std::exp;
    return sigma_sat - (sigma_sat - sigma_y)*exp(-accumulated_plastic_strain/strain_constant);
  };
};

/// @brief J2 material with nonlinear isotropic hardening.
template <typename HardeningType>
struct J2Nonlinear {
  static constexpr int dim = 3;
  static constexpr double tol = 1e-10;

  double E;        ///< Young's modulus
  double nu;       ///< Poisson's ratio
  HardeningType hardening; ///< Flow stress hardening model
  double density;  ///< mass density

  /// @brief variables required to characterize the hysteresis response
  struct State {
    tensor<double, dim, dim> plastic_strain;              ///< plastic strain
    double                   accumulated_plastic_strain;  ///< uniaxial equivalent plastic strain
  };

  /** @brief calculate the Cauchy stress, given the displacement gradient and previous material state */
  template <typename T>
  auto operator()(State& state, const T du_dX) const
  {
    using std::sqrt;
    constexpr auto I = Identity<dim>();
    const double   K = E / (3.0 * (1.0 - 2.0 * nu));
    const double   G = 0.5 * E / (1.0 + nu);

    // (i) elastic predictor
    auto el_strain = sym(du_dX) - state.plastic_strain;
    auto p         = K * tr(el_strain);
    auto s         = 2.0 * G * dev(el_strain);
    auto q = sqrt(1.5)*norm(s);
    
    // (ii) admissibility
    const double eqps_old = state.accumulated_plastic_strain;
    auto residual = [eqps_old, G, *this](auto delta_eqps, auto trial_mises) { return trial_mises - 3.0*G*delta_eqps - this->hardening(eqps_old + delta_eqps);};
    if (residual(0.0, get_value(q)) > tol*hardening.sigma_y) {
      // (iii) return mapping

      // Note the tolerance for convergence is the same as the tolerance for entering the return map.
      // This ensures that if the constitutive update is called again with the updated internal
      // variables, the return map won't be repeated.
      SolverOptions opts{.xtol=0, .rtol=tol*hardening.sigma_y, .max_iter=25};
      double lower_bound = 0.0;
      double upper_bound = (get_value(q) - hardening(eqps_old))/(3.0*G);
      auto [delta_eqps, status] = solve_scalar_equation(residual, 0.0, lower_bound, upper_bound, opts, q);

      auto Np = 1.5*s/q;

      s = s - 2.0 * G * delta_eqps * Np;
      state.accumulated_plastic_strain += get_value(delta_eqps);
      state.plastic_strain += get_value(delta_eqps) * get_value(Np);
    }

    return s + p * I;
  }
};

/// @brief a 3D constitutive model for a J2 material with linear isotropic and kinematic hardening.
struct J2 {
  /// this material is written for 3D
  static constexpr int dim = 3;

  double E;        ///< Young's modulus
  double nu;       ///< Poisson's ratio
  double Hi;       ///< isotropic hardening constant
  double Hk;       ///< kinematic hardening constant
  double sigma_y;  ///< yield stress
  double density;  ///< mass density

  /// @brief variables required to characterize the hysteresis response
  struct State {
    tensor<double, dim, dim> beta;                        ///< back-stress tensor
    tensor<double, dim, dim> plastic_strain;              ///< plastic strain
    double                   accumulated_plastic_strain;  ///< incremental plastic strain
  };

  /** @brief calculate the Cauchy stress, given the displacement gradient and previous material state */
  template <typename T>
  auto operator()(State& state, const T du_dX) const
  {
    using std::sqrt;
    constexpr auto I = Identity<3>();
    const double   K = E / (3.0 * (1.0 - 2.0 * nu));
    const double   G = 0.5 * E / (1.0 + nu);

    //
    // see pg. 260, box 7.5,
    // in "Computational Methods for Plasticity"
    //

    // (i) elastic predictor
    auto el_strain = sym(du_dX) - state.plastic_strain;
    auto p         = K * tr(el_strain);
    auto s         = 2.0 * G * dev(el_strain);
    auto eta       = s - state.beta;
    auto q         = sqrt(3.0 / 2.0) * norm(eta);
    auto phi       = q - (sigma_y + Hi * state.accumulated_plastic_strain);

    // (ii) admissibility
    if (phi > 0.0) {
      // see (7.207) on pg. 261
      auto plastic_strain_inc = phi / (3 * G + Hk + Hi);

      // from here on, only normalize(eta) is required
      // so we overwrite eta with its normalized version
      eta = normalize(eta);

      // (iii) return mapping
      s = s - sqrt(6.0) * G * plastic_strain_inc * eta;
      state.accumulated_plastic_strain += get_value(plastic_strain_inc);
      state.plastic_strain += sqrt(3.0 / 2.0) * get_value(plastic_strain_inc) * get_value(eta);
      state.beta = state.beta + sqrt(2.0 / 3.0) * Hk * get_value(plastic_strain_inc) * get_value(eta);
    }

    return s + p * I;
  }
};

/**
 * @brief Transform the Kirchhoff stress to the Piola stress
 *
 * @tparam T1 number-like type of the displacement gradient components
 * @tparam T1 number-like type of the Kirchhoff stress components
 * @tparam dim number of spatial dimensions
 *
 * @param displacement_gradient Displacement gradient
 * @param kirchhoff_stress Kirchhoff stress
 * @return Piola stress
 */
template <typename T1, typename T2, int dim>
auto KirchhoffToPiola(const tensor<T1, dim, dim>& kirchhoff_stress, const tensor<T2, dim, dim>& displacement_gradient)
{
  return transpose(linear_solve(displacement_gradient + Identity<dim>(), kirchhoff_stress));
}

/**
 * @brief Transform the Cauchy stress to the Piola stress
 *
 * @tparam T1 number-like type of the Cauchy stress components
 * @tparam T2 number-like type of the displacement gradient components
 * @tparam dim number of spatial dimensions
 *
 * @param displacement_gradient Displacement gradient
 * @param cauchy_stress Cauchy stress
 * @return Piola stress
 */
template <typename T1, typename T2, int dim>
auto CauchyToPiola(const tensor<T1, dim, dim>& cauchy_stress, const tensor<T2, dim, dim>& displacement_gradient)
{
  auto kirchhoff_stress = det(displacement_gradient + Identity<dim>()) * cauchy_stress;
  return KirchhoffToPiola(kirchhoff_stress, displacement_gradient);
}

/// Constant body force model
template <int dim>
struct ConstantBodyForce {
  /// The constant body force
  tensor<double, dim> force_;

  /**
   * @brief Evaluation function for the constant body force model
   *
   * @tparam T Position type
   * @tparam dim The dimension of the problem
   * @return The body force value
   */
  template <typename T>
  SERAC_HOST_DEVICE tensor<double, dim> operator()(const tensor<T, dim>& /* x */, const double /* t */) const
  {
    return force_;
  }
};

/// Constant traction boundary condition model
template <int dim>
struct ConstantTraction {
  /// The constant traction
  tensor<double, dim> traction_;

  /**
   * @brief Evaluation function for the constant traction model
   *
   * @tparam T Position type
   * @return The traction value
   */
  template <typename T>
  SERAC_HOST_DEVICE tensor<double, dim> operator()(const tensor<T, dim>& /* x */, const tensor<T, dim>& /* n */,
                                                   const double /* t */) const
  {
    return traction_;
  }
};

}  // namespace serac::solid_mechanics

// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_functional_material.hpp
 *
 * @brief The material and load types for the solid functional physics module
 */

#pragma once

#include "serac/numerics/functional/functional.hpp"

/// SolidFunctional helper data types
namespace serac::solid_mechanics {

/**
 * @brief Linear isotropic elasticity material model
 *
 * @tparam dim Spatial dimension of the mesh
 */
template <int dim>
struct LinearIsotropic {
  using State = Empty;

  /**
   * @brief stress calculation for a linear isotropic material model
   *
   * @tparam DispGradType Displacement gradient type
   * @param du_dX Displacement gradient with respect to the reference configuration
   * @return The calculated material response (density, Kirchoff stress) for the material
   */
  template <typename DispGradType>
  SERAC_HOST_DEVICE auto operator()(State & /* state */, const DispGradType& du_dX) const
  {
    auto I      = Identity<dim>();
    auto lambda = K - (2.0 / dim) * G;
    auto epsilon = 0.5 * (transpose(du_dX) + du_dX);
    return lambda * tr(epsilon) * I + 2.0 * G * epsilon;
  }

  double density;
  double K;
  double G;
};

/**
 * @brief Neo-Hookean material model
 *
 * @tparam dim The spatial dimension of the mesh
 */
template <int dim>
struct NeoHookean {
  using State = Empty;

  /**
   * @brief stress calculation for a NeoHookean material model
   *
   * @tparam DispGradType Displacement gradient type
   * @param displacement_grad displacement gradient with respect to the reference configuration (displacement_grad)
   * @return The calculated material response (density, Kirchoff stress) for the material
   */
  template <typename DispGradType>
  SERAC_HOST_DEVICE auto operator()(State & /* state */, const DispGradType& du_dX) const
  {
    constexpr auto I = Identity<dim>();
    auto lambda = K - (2.0 / dim) * G;
    auto B_minus_I = du_dX * transpose(du_dX) + transpose(du_dX) + du_dX;
    return lambda * log(det(I + du_dX)) * I + G * B_minus_I;
  }

  double density;
  double K;
  double G;
};

/// @brief a 3D constitutive model for a J2 material with linear isotropic and kinematic hardening.
struct J2 {

  static constexpr int dim = 3;

  double E;        ///< Young's modulus
  double nu;       ///< Poisson's ratio
  double Hi;       ///< isotropic hardening constant
  double Hk;       ///< kinematic hardening constant
  double sigma_y;  ///< yield stress
  double density;  ///< mass density

  /// @brief variables required to characterize the hysteresis response
  struct State {
    tensor<double, dim, dim> beta;           ///< back-stress tensor
    tensor<double, dim, dim> el_strain;      ///< elastic strain
    double                   pl_strain;      ///< plastic strain
    double                   pl_strain_inc;  ///< incremental plastic strain
    double                   q;              ///< (trial) J2 stress
  };

  /** @brief calculate the Cauchy stress, given the displacement gradient and previous material state */
  template <typename T>
  auto operator()(State & state, const T du_dX) const
  {
    using std::sqrt;
    constexpr auto I = Identity<3>();
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);

    //
    // see pg. 260, box 7.5,
    // in "Computational Methods for Plasticity"
    //

    // (i) elastic predictor
    auto el_strain = sym(du_dX);
    auto p         = K * tr(el_strain);
    auto s         = 2.0 * G * dev(el_strain);
    auto eta       = s - state.beta;
    auto q         = sqrt(3.0 / 2.0) * norm(eta);
    auto phi       = q - (sigma_y + Hi * state.pl_strain);

    // (ii) admissibility
    if (phi > 0.0) {

      // std::cout << "returning stress to yield surface" << std::endl;

      // see (7.207) on pg. 261
      auto plastic_strain_inc = phi / (3 * G + Hk + Hi);

      // (iii) return mapping
      s = s - sqrt(6.0) * G * plastic_strain_inc * normalize(eta);

      state.pl_strain = state.pl_strain + get_value(plastic_strain_inc);

      state.beta = state.beta + sqrt(2.0 / 3.0) * Hk * get_value(plastic_strain_inc) * normalize(get_value(eta));
    }

    return s + p * I;
  }

};

/// Constant body force model
template <int dim>
struct ConstantBodyForce {
  /// The constant body force
  tensor<double, dim> force_;

  /**
   * @brief Evaluation function for the constant body force model
   *
   * @tparam DisplacementType Displacement type
   * @tparam DispGradType Displacement gradient type
   * @tparam dim The dimension of the problem
   * @return The body force value
   */
  SERAC_HOST_DEVICE tensor<double, dim> operator()(const tensor<double, dim>& /* x */, const double /* t */) const
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
   * @return The traction value
   */
  SERAC_HOST_DEVICE tensor<double, dim> operator()(const tensor<double, dim>& /* x */,
                                                   const tensor<double, dim>& /* n */, const double /* t */) const
  {
    return traction_;
  }
};

/// Function-based traction boundary condition model
template <int dim>
struct TractionFunction {
  /// The traction function
  std::function<tensor<double, dim>(const tensor<double, dim>&, const tensor<double, dim>&, const double)>
      traction_func_;

  /**
   * @brief Evaluation for the function-based traction model
   *
   * @param x The spatial coordinate
   * @param n The normal vector
   * @param t The current time
   * @return The traction to apply
   */
  SERAC_HOST_DEVICE tensor<double, dim> operator()(const tensor<double, dim>& x, const tensor<double, dim>& n,
                                                   const double t) const
  {
    return traction_func_(x, n, t);
  }
};

/// Constant pressure model
struct ConstantPressure {
  /// The constant pressure
  double pressure_;

  /**
   * @brief Evaluation of the constant pressure model
   *
   * @tparam dim Spatial dimension
   */
  template <int dim>
  SERAC_HOST_DEVICE double operator()(const tensor<double, dim>& /* x */, const double /* t */) const
  {
    return pressure_;
  }
};

/// Function-based pressure boundary condition
template <int dim>
struct PressureFunction {
  /// The pressure function
  std::function<double(const tensor<double, dim>&, const double)> pressure_func_;

  /**
   * @brief Evaluation for the function-based pressure model
   *
   * @param x The spatial coordinate
   * @param t The current time
   * @return The pressure to apply
   */
  SERAC_HOST_DEVICE double operator()(const tensor<double, dim>& x, const double t) const
  {
    return pressure_func_(x, t);
  }
};

}  // namespace serac::solid_util

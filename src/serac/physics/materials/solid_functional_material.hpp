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
namespace serac::solid_util {

/**
 * @brief Response data type for solid mechanics simulations
 *
 * @tparam T1 Density type
 * @tparam T2 Stress type (i.e. second order tensor)
 */
template <typename T1, typename T2>
struct MaterialResponse {
  /// Density of the material (mass/volume)
  T1 density;

  /// Kirchoff stress (det(deformation gradient) * Cauchy stress) for the constitutive model
  T2 stress;
};

/**
 * @brief Template deduction guide for the material response
 *
 * @tparam T1 Density type
 * @tparam T2 Stress type
 */
template <typename T1, typename T2>
MaterialResponse(T1, T2) -> MaterialResponse<T1, T2>;

/**
 * @brief Linear isotropic elasticity material model
 *
 * @tparam dim Spatial dimension of the mesh
 */
template <int dim>
class LinearIsotropicSolid {
public:
  /**
   * @brief Construct a new Linear Isotropic Elasticity object
   *
   * @param density Density of the material
   * @param shear_modulus Shear modulus of the material
   * @param bulk_modulus Bulk modulus of the material
   */
  LinearIsotropicSolid(double density = 1.0, double shear_modulus = 1.0, double bulk_modulus = 1.0)
      : density_(density), bulk_modulus_(bulk_modulus), shear_modulus_(shear_modulus)
  {
    SLIC_ERROR_ROOT_IF(shear_modulus_ < 0.0,
                       "Shear modulus must be positive in the linear isotropic elasticity material model.");

    SLIC_ERROR_ROOT_IF(density_ < 0.0, "Density must be positive in the linear isotropic elasticity material model.");

    SLIC_ERROR_ROOT_IF(bulk_modulus_ < 0.0,
                       "Bulk modulus must be positive in the linear isotropic elasticity material model.");

    double K             = bulk_modulus;
    double G             = shear_modulus;
    double poisson_ratio = (3 * K - 2 * G) / (6 * K + 2 * G);

    SLIC_ERROR_ROOT_IF(poisson_ratio < 0.0,
                       "Poisson ratio must be positive in the linear isotropic elasticity material model.");
  }

  /**
   * @brief Material response call for a linear isotropic solid
   *
   * @tparam T1 Spatial position type
   * @tparam T2 Displacement type
   * @tparam T3 Displacement gradient type
   * @param du_dX displacement gradient with respect to the reference configuration (du_dX)
   * @return The calculated material response (density, kirchoff stress) for the material
   */
  template <typename T1, typename T2, typename T3>
  SERAC_HOST_DEVICE auto operator()(const T1& /* x */, const T2& /* displacement */, const T3& du_dX) const
  {
    auto I      = Identity<dim>();
    auto lambda = bulk_modulus_ - (2.0 / dim) * shear_modulus_;
    auto strain = 0.5 * (du_dX + transpose(du_dX));
    auto stress = lambda * tr(strain) * I + 2.0 * shear_modulus_ * strain;
    return MaterialResponse<double, T3>{.density = density_, .stress = stress};
  }

private:
  /// Density
  double density_;

  /// Bulk modulus
  double bulk_modulus_;

  /// Shear modulus
  double shear_modulus_;
};

/**
 * @brief Neo-Hookean material model
 *
 * @tparam dim The spatial dimension of the mesh
 */
template <int dim>
class NeoHookeanSolid {
public:
  /**
   * @brief Construct a new Neo-Hookean object
   *
   * @param density Density of the material
   * @param shear_modulus Shear modulus of the material
   * @param bulk_modulus Bulk modulus of the material
   */
  NeoHookeanSolid(double density = 1.0, double shear_modulus = 1.0, double bulk_modulus = 1.0)
      : density_(density), bulk_modulus_(bulk_modulus), shear_modulus_(shear_modulus)
  {
    SLIC_ERROR_ROOT_IF(shear_modulus_ < 0.0, "Shear modulus must be positive in the neo-Hookean material model.");

    SLIC_ERROR_ROOT_IF(density_ < 0.0, "Density must be positive in the neo-Hookean material model.");

    SLIC_ERROR_ROOT_IF(bulk_modulus_ < 0.0, "Bulk modulus must be positive in the neo-Hookean material model.");

    double K             = bulk_modulus;
    double G             = shear_modulus;
    double poisson_ratio = (3 * K - 2 * G) / (6 * K + 2 * G);

    SLIC_ERROR_ROOT_IF(poisson_ratio < 0.0, "Poisson ratio must be positive in the neo-Hookean material model.");
  }

  /**
   * @brief Material response call for a neo-Hookean solid
   *
   * @tparam T1 Spatial position type
   * @tparam T2 Displacement type
   * @tparam T3 Displacement gradient type
   * @param du_dX displacement gradient with respect to the reference configuration (du_dX)
   * @return The calculated material response (density, kirchoff stress) for the material
   */
  template <typename T1, typename T2, typename T3>
  SERAC_HOST_DEVICE auto operator()(const T1& /* x */, const T2& /* displacement */, const T3& du_dX) const
  {
    auto I         = Identity<dim>();
    auto lambda    = bulk_modulus_ - (2.0 / dim) * shear_modulus_;
    auto B_minus_I = du_dX * transpose(du_dX) + transpose(du_dX) + du_dX;

    auto J = det(du_dX + I);

    // TODO this resolve to the correct std implementation of log when J resolves to a pure double. It can
    // be removed by either putting the dual implementation of the global namespace or implementing a pure
    // double version there. More investigation into argument-dependent lookup is needed.
    using std::log;
    auto stress = lambda * log(J) * I + shear_modulus_ * B_minus_I;
    return MaterialResponse<double, T3>{.density = density_, .stress = stress};
  }

private:
  /// Density
  double density_;

  /// Bulk modulus in the stress free configuration
  double bulk_modulus_;

  /// Shear modulus in the stress free configuration
  double shear_modulus_;
};

/// Constant body force model
template <int dim>
struct ConstantBodyForce {
  /// The constant body force
  tensor<double, dim> force_;

  /**
   * @brief Evaluation function for the constant body force model
   *
   * @tparam T1 type of the displacement
   * @tparam T2 type of the displacement gradient
   * @tparam dim The dimension of the problem
   * @return The body force value
   */
  template <typename T1, typename T2>
  SERAC_HOST_DEVICE tensor<double, dim> operator()(const tensor<double, dim>& /* x */, const double /* t */,
                                                   const T1& /* u */, const T2& /* du_dX */) const
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
  std::function<tensor<double, dim>(const tensor<double, dim>&, const tensor<double, dim>&, const double t)>
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

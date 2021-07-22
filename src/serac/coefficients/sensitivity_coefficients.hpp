// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file sensitivity_coefficients.hpp
 *
 * @brief Coefficients to help with computing sensitivities
 */

#pragma once

#include "serac/physics/utilities/finite_element_state.hpp"
#include "serac/physics/materials/hyperelastic_material.hpp"
#include "serac/coefficients/sensitivity_coefficients.hpp"

#include "mfem.hpp"

namespace serac::mfem_ext {

/**
 * @brief A coefficient containing the pseudo-load (explicit derivative of the residual) for the shear modulus
 *
 */
class ShearSensitivityCoefficient : public mfem::Coefficient {
public:
  /**
   * @brief Construct a new Shear Sensitivity Coefficient object
   *
   * @param displacement The displacement state for computing the sensitivities via the adjoint method
   * @param adjoint_displacement The adjoint state for computing the sensitivities via the adjoint method
   * @param linear_mat The linear elastic material model
   */
  ShearSensitivityCoefficient(FiniteElementState& displacement, FiniteElementState& adjoint_displacement,
                              LinearElasticMaterial& linear_mat);

  /**
   * @brief Do not allow default construction of the shear sensitivity coefficient
   *
   */
  ShearSensitivityCoefficient() = delete;

  /**
   * @brief Evaluate the sensitivity coefficient at an integration point
   *
   * @param T The element transformation
   * @param ip The integration point
   * @return The pseudoload (explicit derivative of the residual) wrt the shear modulus
   */
  virtual double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip);

private:
  /**
   * @brief The displacement state for computing the sensitivities via the adjoint method
   *
   */
  FiniteElementState& displacement_;

  /**
   * @brief The adjoint state for computing the sensitivities via the adjoint method
   *
   */
  FiniteElementState& adjoint_displacement_;

  /**
   * @brief The linear elastic material model
   *
   */
  LinearElasticMaterial& material_;

  /**
   * @brief The displacement gradient
   *
   */
  mfem::DenseMatrix du_dX_;

  /**
   * @brief The adjoint variable gradient
   *
   */
  mfem::DenseMatrix dp_dX_;

  /**
   * @brief The derivative of the Cauchy stress wrt the shear modulus
   *
   */
  mutable mfem::DenseMatrix d_sigma_d_shear_;

  /**
   * @brief Linearized strain of the adjoint field
   *
   */
  mutable mfem::DenseMatrix adjoint_strain_;

  /**
   * @brief The deformation gradient (dx/dX)
   *
   */
  mutable mfem::DenseMatrix F_;

  /**
   * @brief Volumetric dimension of the problem
   *
   */
  int dim_;
};

/**
 * @brief A coefficient containing the pseudo-load (explicit derivative of the residual) for the bulk modulus
 *
 */
class BulkSensitivityCoefficient : public mfem::Coefficient {
public:
  /**
   * @brief Construct a new Bulk Sensitivity Coefficient object
   *
   * @param displacement The displacement state for computing the sensitivities via the adjoint method
   * @param adjoint_displacement The adjoint state for computing the sensitivities via the adjoint method
   * @param linear_mat The linear elastic material model
   */
  BulkSensitivityCoefficient(FiniteElementState& displacement, FiniteElementState& adjoint_displacement,
                             LinearElasticMaterial& linear_mat);

  /**
   * @brief Do not allow default construction of the bulk sensitivity coefficient
   *
   */
  BulkSensitivityCoefficient() = delete;

  /**
   * @brief Evaluate the sensitivity coefficient at an integration point
   *
   * @param T The element transformation
   * @param ip The integration point
   * @return The pseudoload (explicit derivative of the residual) wrt the bulk modulus
   */
  virtual double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip);

private:
  /**
   * @brief The displacement state for computing the sensitivities via the adjoint method
   *
   */
  FiniteElementState& displacement_;

  /**
   * @brief The adjoint state for computing the sensitivities via the adjoint method
   *
   */
  FiniteElementState& adjoint_displacement_;

  /**
   * @brief The linear elastic material model
   *
   */
  LinearElasticMaterial& material_;

  /**
   * @brief The displacement gradient
   *
   */
  mfem::DenseMatrix du_dX_;

  /**
   * @brief The adjoint variable gradient
   *
   */
  mfem::DenseMatrix dp_dX_;

  /**
   * @brief The derivative of the Cauchy stress wrt the bulk modulus
   *
   */
  mfem::DenseMatrix d_sigma_d_bulk_;

  /**
   * @brief Linearized strain of the adjoint field
   *
   */
  mfem::DenseMatrix adjoint_strain_;

  /**
   * @brief The deformation gradient (dx/dX)
   *
   */
  mfem::DenseMatrix F_;

  /**
   * @brief Volumetric dimension of the problem
   *
   */
  int dim_;
};

}  // namespace serac::mfem_ext

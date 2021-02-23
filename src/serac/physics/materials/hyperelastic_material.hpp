// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file hyperelastic_material.hpp
 *
 * @brief The hyperelastic material models for the solid module
 */

#pragma once

#include "mfem.hpp"

#include "serac/numerics/array_4D.hpp"

namespace serac {

/**
 * @brief Abstract interface class for a generic hyperelastic material
 *
 */
class HyperelasticMaterial {
public:
  /**
   * @brief Construct a new Hyperelastic Material object
   *
   */
  HyperelasticMaterial() : parent_to_reference_transformation_(nullptr) {}

  /**
   * @brief Destroy the Hyperelastic Material object
   *
   */
  virtual ~HyperelasticMaterial() = default;

  /// A parent-element to reference-element transformation that can be used to
  /// evaluate mfem::Coefficient%s.
  /** @note It is assumed that parent_to_reference_transformation_.SetIntPoint() is already called for the
      point of interest. */

  /**
   * @brief Set the reference-to-target transformation. This is required to use coefficient parameters.
   *
   * @param[in] parent_to_reference_transformation_ The reference-to-target (stress-free) transformation
   */
  void setTransformation(mfem::ElementTransformation& Ttr) { parent_to_reference_transformation_ = &Ttr; }

  /**
   * @brief Evaluate the strain energy density function, W = W(F).
   *
   * @param[in] du_dX the displacement gradient
   * @return double Strain energy density
   */
  virtual double evalStrainEnergy(const mfem::DenseMatrix& du_dX) const = 0;

  /**
   * @brief Evaluate the Cauchy stress sigma = sigma(F).
   *
   * @param[in] du_dX the displacement gradient
   * @param[out] sigma The evaluated Cauchy stress
   */
  virtual void evalStress(const mfem::DenseMatrix& du_dX, mfem::DenseMatrix& sigma) const = 0;

  /**
   * @brief Evaluate the derivative of the 1st Piola-Kirchhoff stress tensor in Voigt notation
   * and assemble its contribution to the local gradient matrix 'A'.

   * @param[in] du_dX the displacement gradient
   * @param[out] C Tangent moduli 4D Array in spatial form (C^e_ijkl=(d tau_ij)/(d F_km) * F_lm = J * sigma_ij delta_kl
   + J * (d sigma_ij)/(d F_km) F_lm )
   */
  virtual void evalTangentStiffness(const mfem::DenseMatrix& du_dX, serac::mfem_ext::Array4D<double>& C) const = 0;

protected:
  /**
   * @brief Reference element to stree-free configuration (target) transformation
   *
   */
  mfem::ElementTransformation* parent_to_reference_transformation_;
};

/**
 * @brief Neo-Hookean hyperelastic model with a strain energy density function given
 *   by the formula: \f$(\mu/2)(\bar{I}_1 - dim) + (K/2)(det(F)/g - 1)^2\f$ where
 *   F is the deformation gradient and \f$\bar{I}_1 = (det(F))^{-2/dim} Tr(F
 *   F^t)\f$. The parameters \f$\mu\f$ and \f$\lambda\f$ are the Lame parameters.
 *
 */
class NeoHookeanMaterial : public HyperelasticMaterial {
public:
  /**
   * @brief Construct a new Neo Hookean Material object
   *
   * @param[in] mu Shear modulus mu
   * @param[in] bulk Bulk modulus K
   */
  NeoHookeanMaterial(std::unique_ptr<mfem::Coefficient>&& mu, std::unique_ptr<mfem::Coefficient>&& bulk)
      : c_mu_(std::move(mu)), c_bulk_(std::move(bulk))
  {
  }

  /**
   * @brief Evaluate the strain energy density function, W = W(F).
   *
   * @param[in] du_dX The displacement gradient
   * @return Strain energy density
   */
  virtual double evalStrainEnergy(const mfem::DenseMatrix& du_dX) const;

  /**
   * @brief Evaluate the Cauchy stress
   *
   * @param[in] du_dX The displacement gradient
   * @param[out] sigma The evaluated Cauchy stress
   */
  virtual void evalStress(const mfem::DenseMatrix& du_dX, mfem::DenseMatrix& sigma) const;

  /**
   * @brief Evaluate the derivative of the Kirchoff stress wrt the deformation gradient
   * and assemble its contribution to the 4D array (spatial elasticity tensor)
   * @param[in] du_dX The displacement gradient
   * @param[out] C Tangent moduli 4D Array
   */
  virtual void evalTangentStiffness(const mfem::DenseMatrix& du_dX, serac::mfem_ext::Array4D<double>& C) const;

  /**
   * @brief Destroy the Hyperelastic Material object
   *
   */
  virtual ~NeoHookeanMaterial() = default;

  /**
   * @brief Disable the default constructor
   *
   */
  NeoHookeanMaterial() = delete;

protected:
  /**
   * @brief Shear modulus in constant form
   *
   */
  mutable double mu_;

  /**
   * @brief Bulk modulus in constant form
   *
   */
  mutable double bulk_;

  /**
   * @brief Shear modulus in coefficient form
   *
   */
  std::unique_ptr<mfem::Coefficient> c_mu_;

  /**
   * @brief Bulk modulus in coefficient form
   *
   */
  std::unique_ptr<mfem::Coefficient> c_bulk_;

  /**
   * @brief The deformation gradient (dx_dX)
   *
   */
  mutable mfem::DenseMatrix F_;

  /**
   * @brief The left Cauchy-Green deformation tensor (FF^T)
   *
   */
  mutable mfem::DenseMatrix B_;

  /**
   * @brief Evaluate the coefficients
   * @note The reference-to-target transformation must be set before this call.
   *
   */
  inline void EvalCoeffs() const;
};

/**
 * @brief Linear elastic material model
 *
 */
class LinearElasticMaterial : public HyperelasticMaterial {
public:
  /**
   * @brief Construct a new Linear Elastic Material object
   *
   * @param[in] mu Shear modulus mu
   * @param[in] bulk Bulk modulus K
   */
  LinearElasticMaterial(std::unique_ptr<mfem::Coefficient>&& mu, std::unique_ptr<mfem::Coefficient>&& bulk)
      : c_mu_(std::move(mu)), c_bulk_(std::move(bulk))
  {
  }

  /**
   * @brief Evaluate the strain energy density function, W = W(F).
   *
   * @param[in] du_dX the displacement gradient
   * @return Strain energy density
   */
  virtual double evalStrainEnergy(const mfem::DenseMatrix&) const
  {
    SLIC_ERROR("Strain energy not implemented for the linear elastic material!");
    return 0.0;
  }

  /**
   * @brief Evaluate the Cauchy stress
   *
   * @param[in] du_dX the displacement gradient
   * @param[out] sigma The evaluated Cauchy stress
   */
  virtual void evalStress(const mfem::DenseMatrix& du_dX, mfem::DenseMatrix& sigma) const;

  /**
   * @brief Evaluate the derivative of the Kirchoff stress wrt the deformation gradient
   * and assemble its contribution to the 4D array (spatial elasticity tensor)
   * @param[in] du_dX the displacement gradient
   * @param[out] C Tangent moduli 4D Array
   */
  virtual void evalTangentStiffness(const mfem::DenseMatrix& du_dX, serac::mfem_ext::Array4D<double>& C) const;

  /**
   * @brief Destroy the Hyperelastic Material object
   *
   */
  virtual ~LinearElasticMaterial() = default;

  /**
   * @brief Disable the default constructor
   *
   */
  LinearElasticMaterial() = delete;

protected:
  /**
   * @brief Shear modulus in constant form
   *
   */
  mutable double mu_;

  /**
   * @brief Bulk modulus in constant form
   *
   */
  mutable double bulk_;

  /**
   * @brief Shear modulus in coefficient form
   *
   */
  std::unique_ptr<mfem::Coefficient> c_mu_;

  /**
   * @brief Bulk modulus in coefficient form
   *
   */
  std::unique_ptr<mfem::Coefficient> c_bulk_;

  /**
   * @brief The linearized strain tensor
   *
   */
  mutable mfem::DenseMatrix epsilon_;

  /**
   * @brief Evaluate the coefficients
   * @note The reference-to-target transformation must be set before this call.
   *
   */
  inline void EvalCoeffs() const;
};

}  // namespace serac

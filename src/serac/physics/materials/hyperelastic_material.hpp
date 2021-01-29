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

#ifndef HYPERELAS
#define HYPERELAS

#include "mfem.hpp"

#include "serac/numerics/voigt_tensor.hpp"

namespace serac {

/**
 * @brief Abstract interface class for a generic hyperelastic material
 *
 */
class HyperelasticMaterial {
protected:
  /**
   * @brief Reference element to stree-free configuration (target) transformation
   *
   */
  mfem::ElementTransformation* Ttr_;

  /**
   * @brief The shear terms needed for Voigt tensor notation
   */
  std::vector<std::pair<int, int>> shear_terms_;

public:
  /**
   * @brief Construct a new Hyperelastic Material object
   *
   * @param[in] dim The dimension of the problem
   */
  HyperelasticMaterial(const int dim) : Ttr_(nullptr) { getShearTerms(dim, shear_terms_); }

  /**
   * @brief Destroy the Hyperelastic Material object
   *
   */
  virtual ~HyperelasticMaterial() {}

  /// A reference-element to target-element transformation that can be used to
  /// evaluate mfem::Coefficient%s.
  /** @note It is assumed that _Ttr.SetIntPoint() is already called for the
      point of interest. */

  /**
   * @brief Set the reference-to-target transformation. This is required to use coefficient parameters.
   *
   * @param[in] Ttr The reference-to-target (stress-free) transformation
   */
  void SetTransformation(mfem::ElementTransformation& Ttr) { Ttr_ = &Ttr; }

  /** @brief
      @param[in] C  Right Cauchy-Green Deformation Tensor (F^T F) */

  /**
   * @brief Evaluate the strain energy density function, W = W(C).
   *
   * @param[in] C Right Cauchy-Green Deformation Tensor (F^T F)
   * @return double Strain energy density
   */
  virtual double EvalW(const mfem::DenseMatrix& C) const = 0;

  /**
   * @brief Evaluate the 2nd Piola-Kirchhoff stress tensor, S = S(C).
   *
   * @param[in] C Right Cauchy-Green Deformation Tensor (F^T F)
   * @param[out] S The evaluated 2nd Piola-Kirchhoff stress tensor in Voigt notation.
   */
  virtual void EvalPK2(const mfem::DenseMatrix& C, mfem::Vector& S) const = 0;

  /**
   * @brief Evaluate the derivative of the 1st Piola-Kirchhoff stress tensor in Voigt notation
   * and assemble its contribution to the local gradient matrix 'A'.
   * @param[in] C     Right Cauchy-Green Deformation Tensor (F^T F)
   * @param[out] T  Tangent moduli matrix in Voigt notation
   */
  virtual void AssembleTangentModuli(const mfem::DenseMatrix& C, mfem::DenseMatrix& T) const = 0;
};

/**
 * @brief Neo-Hookean hyperelastic model with a strain energy density function given
 *   by the formula: \f$(\mu/2)(\bar{I}_1 - dim) + (K/2)(det(F)/g - 1)^2\f$ where
 *   F is the deformation gradient and \f$\bar{I}_1 = (det(F))^{-2/dim} Tr(F
 *   F^t)\f$. The parameters \f$\mu\f$ and \f$\lambda\f$ are the Lame parameters.
 *
 */
class NeoHookeanMaterial : public HyperelasticMaterial {
protected:
  /**
   * @brief Lame parameters in constant form
   *
   */
  mutable double mu_, lambda_;

  /**
   * @brief Shear and bulk modulus in coefficient form
   *
   */
  mfem::Coefficient *c_mu_, *c_bulk_;

  /**
   * @brief The inverse of the right Cauchy-Green deformation tensor
   *
   */
  mutable mfem::DenseMatrix Cinv_;

  /**
   * @brief The identity matrix
   *
   */
  mutable mfem::DenseMatrix eye_;

  /**
   * @brief The 2nd Piola-Kirchoff stress
   *
   */
  mutable mfem::DenseMatrix S_;

  /**
   * @brief A vector of pairs of shear terms for Voigt notation
   *
   */
  mutable std::vector<std::pair<int, int>> shear_terms_;

  /**
   * @brief Evaluate the coefficient.
   * @note The reference-to-target transformation must be set before this call.
   *
   */
  inline void EvalCoeffs() const;

public:
  /**
   * @brief Construct a new Neo Hookean Material object
   *
   * @param[in] dim Dimension of the problem
   * @param[in] mu Shear modulus
   * @param[in] bulk Bulk modulus
   */
  NeoHookeanMaterial(const int dim, double mu, double bulk)
      : HyperelasticMaterial(dim), mu_(mu), lambda_(bulk - (2.0 / 3.0) * mu)
  {
    c_mu_   = nullptr;
    c_bulk_ = nullptr;

    getShearTerms(dim, shear_terms_);

    eye_.SetSize(dim);
    eye_ = 0.0;
    for (int i = 0; i < dim; ++i) {
      eye_(i, i) = 1.0;
    }
  }

  /**
   * @brief Construct a new Neo Hookean Material object
   *
   * @param[in] dim Dimension of the problem
   * @param[in] mu Shear modulus mu
   * @param[in] bulk Bulk modulus K
   */
  NeoHookeanMaterial(const int dim, mfem::Coefficient& mu, mfem::Coefficient& bulk)
      : HyperelasticMaterial(dim), mu_(0.0), lambda_(0.0), c_mu_(&mu), c_bulk_(&bulk)
  {
    getShearTerms(dim, shear_terms_);

    eye_.SetSize(dim);
    eye_ = 0.0;
    for (int i = 0; i < dim; ++i) {
      eye_(i, i) = 1.0;
    }
  }

  /** @brief
      @param[in] C  Right Cauchy-Green Deformation Tensor (F^T F) */

  /**
   * @brief Evaluate the strain energy density function, W = W(C).
   *
   * @param[in] C Right Cauchy-Green Deformation Tensor (F^T F)
   * @return double Strain energy density
   */
  virtual double EvalW(const mfem::DenseMatrix& C) const;

  /**
   * @brief Evaluate the 2nd Piola-Kirchhoff stress tensor, S = S(C).
   *
   * @param[in] C Right Cauchy-Green Deformation Tensor (F^T F)
   * @param[out] S The evaluated 2nd Piola-Kirchhoff stress tensor in Voigt notation.
   */
  virtual void EvalPK2(const mfem::DenseMatrix& C, mfem::Vector& S) const;

  /**
   * @brief Evaluate the derivative of the 1st Piola-Kirchhoff stress tensor in Voigt notation
   * and assemble its contribution to the local gradient matrix 'A'.
   * @param[in] C     Right Cauchy-Green Deformation Tensor (F^T F)
   * @param[out] T  Tangent moduli matrix in Voigt notation
   */
  virtual void AssembleTangentModuli(const mfem::DenseMatrix& C, mfem::DenseMatrix& T) const;
};

}  // namespace serac

#endif
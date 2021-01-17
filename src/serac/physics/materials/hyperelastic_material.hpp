// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
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

namespace serac {

inline void getShearTerms(const int dim, std::vector<std::pair<int, int>>& shear_terms)
{
  if (shear_terms.size() == 0) {
    if (dim == 2) {
      shear_terms = {{1, 2}};
    } else {
      shear_terms = {{2, 3}, {1, 3}, {1, 2}};
    }
  }
}

void getTensorFromVoigtVector(const std::vector<std::pair<int, int>>& shear_terms, const mfem::Vector& vec,
                              mfem::DenseMatrix& mat);

void getVoigtVectorFromTensor(const std::vector<std::pair<int, int>>& shear_terms, const mfem::DenseMatrix& mat,
                              mfem::Vector& vec);

/// Abstract class for hyperelastic models
class HyperelasticMaterial {
protected:
  mfem::ElementTransformation* Ttr_; /**< Reference-element to target-element
                                   transformation. */

  /**
   * @brief The shear terms needed for Voigt tensor notation
   */
  std::vector<std::pair<int, int>> shear_terms_;

public:
  HyperelasticMaterial(const int dim) : Ttr_(nullptr) { getShearTerms(dim, shear_terms_); }
  virtual ~HyperelasticMaterial() {}

  /// A reference-element to target-element transformation that can be used to
  /// evaluate mfem::Coefficient%s.
  /** @note It is assumed that _Ttr.SetIntPoint() is already called for the
      point of interest. */
  void SetTransformation(mfem::ElementTransformation& Ttr) { Ttr_ = &Ttr; }

  /** @brief Evaluate the strain energy density function, W = W(C).
      @param[in] C  Right Cauchy-Green Deformation Tensor (F^T F) */
  virtual double EvalW(const mfem::DenseMatrix& C) const = 0;

  /** @brief Evaluate the 2nd Piola-Kirchhoff stress tensor, S = S(C).
      @param[in] C  Right Cauchy-Green Deformation Tensor (F^T F)
      @param[out]  S  The evaluated 1st Piola-Kirchhoff stress tensor in Voigt notation. */
  virtual void EvalPK2(const mfem::DenseMatrix& C, mfem::Vector& S) const = 0;

  /** @brief Evaluate the derivative of the 1st Piola-Kirchhoff stress tensor in Voigt notation
      and assemble its contribution to the local gradient matrix 'A'.
      @param[in] C     Right Cauchy-Green Deformation Tensor (F^T F)
      @param[in,out]  T  Tangent moduli matrix in Voigt notation
  */
  virtual void AssembleTangentModuli(const mfem::DenseMatrix& C, mfem::DenseMatrix& T) const = 0;
};

/** Neo-Hookean hyperelastic model with a strain energy density function given
    by the formula: \f$(\mu/2)(\bar{I}_1 - dim) + (K/2)(det(F)/g - 1)^2\f$ where
    F is the deformation gradient and \f$\bar{I}_1 = (det(F))^{-2/dim} Tr(F
    F^t)\f$. The parameters \f$\mu\f$ and K are the shear and bulk moduli,
    respectively. */
class NeoHookeanMaterial : public HyperelasticMaterial {
protected:
  mutable double            mu_, lambda_;
  mfem::Coefficient *       c_mu_, *c_lambda_;
  mutable mfem::DenseMatrix Cinv_;
  mutable mfem::DenseMatrix eye_;
  mutable mfem::DenseMatrix S_;

  mutable std::vector<std::pair<int, int>> shear_terms_;

  inline void EvalCoeffs() const;

public:
  NeoHookeanMaterial(const int dim, double mu, double lambda) : HyperelasticMaterial(dim), mu_(mu), lambda_(lambda)
  {
    c_mu_     = nullptr;
    c_lambda_ = nullptr;
  }

  NeoHookeanMaterial(const int dim, mfem::Coefficient& mu, mfem::Coefficient& lambda)
      : HyperelasticMaterial(dim), mu_(0.0), lambda_(0.0), c_mu_(&mu), c_lambda_(&lambda)
  {
  }

  virtual double EvalW(const mfem::DenseMatrix& C) const;

  virtual void EvalPK2(const mfem::DenseMatrix& C, mfem::Vector& S) const;

  virtual void AssembleTangentModuli(const mfem::DenseMatrix& C, mfem::DenseMatrix& T) const;
};

}  // namespace serac

#endif
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

/// Abstract class for hyperelastic models
class HyperelasticMaterial {
protected:
  mfem::ElementTransformation* Ttr; /**< Reference-element to target-element
                                   transformation. */

public:
  HyperelasticMaterial() : Ttr(NULL) {}
  virtual ~HyperelasticMaterial() {}

  /// A reference-element to target-element transformation that can be used to
  /// evaluate mfem::Coefficient%s.
  /** @note It is assumed that _Ttr.SetIntPoint() is already called for the
      point of interest. */
  void SetTransformation(mfem::ElementTransformation& _Ttr) { Ttr = &_Ttr; }

  /** @brief Evaluate the strain energy density function, W = W(F).
      @param[in] F  Represents the target->physical transformation
                      Jacobian matrix. */
  virtual double EvalW(const mfem::DenseMatrix& F) const = 0;

  /** @brief Evaluate the 1st Piola-Kirchhoff stress tensor, P = P(F).
      @param[in] F  Represents the target->physical transformation
                      Jacobian matrix.
      @param[out]  P  The evaluated 1st Piola-Kirchhoff stress tensor. */
  virtual void EvalP(const mfem::DenseMatrix& F, mfem::DenseMatrix& P) const = 0;

  /** @brief Evaluate the derivative of the 1st Piola-Kirchhoff stress tensor
      and assemble its contribution to the local gradient matrix 'A'.
      @param[in] F     Represents the target->physical transformation
                         Jacobian matrix.
      @param[in] BO_T      Gradient of the basis matrix (dof x dim).
      @param[in] weight  Quadrature weight mfem::Coefficient for the point.
      @param[in,out]  A  Local gradient matrix where the contribution from this
                         point will be added.
      Computes weight * d(dW_dxi)_d(xj) at the current point, for all i and j,
      where x1 ... xn are the FE dofs. This function is usually defined using
      the matrix invariants and their derivatives.
  */
  virtual void AssembleTangentModuli(const mfem::DenseMatrix& F, const mfem::DenseMatrix& B0_T, const double weight,
                                     mfem::DenseMatrix& C) const = 0;
};

/** Neo-Hookean hyperelastic model with a strain energy density function given
    by the formula: \f$(\mu/2)(\bar{I}_1 - dim) + (K/2)(det(F)/g - 1)^2\f$ where
    F is the deformation gradient and \f$\bar{I}_1 = (det(F))^{-2/dim} Tr(F
    F^t)\f$. The parameters \f$\mu\f$ and K are the shear and bulk moduli,
    respectively, and g is a reference volumetric scaling. */
class NeoHookeanMaterial : public HyperelasticMaterial {
protected:
  mutable double     mu, K, g;
  mfem::Coefficient *c_mu, *c_K, *c_g;
  bool               have_coeffs;

  mutable mfem::DenseMatrix FinvT;  // dim x dim
  mutable mfem::DenseMatrix G, C;   // dof x dim

  inline void EvalCoeffs() const;

public:
  NeoHookeanMaterial(double _mu, double _K, double _g = 1.0) : mu(_mu), K(_K), g(_g), have_coeffs(false)
  {
    c_mu = c_K = c_g = NULL;
  }

  NeoHookeanMaterial(mfem::Coefficient& _mu, mfem::Coefficient& _K, mfem::Coefficient* _g = NULL)
      : mu(0.0), K(0.0), g(1.0), c_mu(&_mu), c_K(&_K), c_g(_g), have_coeffs(true)
  {
  }

  virtual double EvalW(const mfem::DenseMatrix& F) const;

  virtual void EvalP(const mfem::DenseMatrix& F, mfem::DenseMatrix& P) const;

  virtual void AssembleTangentModuli(const mfem::DenseMatrix& F, const mfem::DenseMatrix& B0_T, const double weight,
                                     mfem::DenseMatrix& C) const;
};

}  // namespace serac

#endif
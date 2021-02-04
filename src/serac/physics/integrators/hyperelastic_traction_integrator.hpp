// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file hyperelastic_traction_integrator.hpp
 *
 * @brief Custom MFEM integrator for nonlinear finite deformation traction loads
 */

#pragma once

#include "mfem.hpp"

namespace serac::mfem_ext {

/**
 * @brief Custom MFEM integrator for nonlinear finite deformation traction loads
 */
class HyperelasticTractionIntegrator : public mfem::NonlinearFormIntegrator {
public:
  /**
   * @brief Construct a new Hyperelastic Traction Integrator object
   *
   * @param[in] f The traction load coefficient
   * @param[in] compute_on_reference Flag for computing the residual in the stress-free configuration vs. the deformed
   * configuration
   */
  explicit HyperelasticTractionIntegrator(mfem::VectorCoefficient& f, bool compute_on_reference = false)
      : function_(f), compute_on_reference_(compute_on_reference)
  {
  }

  /**
   * @brief Assemble the nonlinear residual on a boundary element at a current state
   *
   * @param[in] el1 The first element attached to the face
   * @param[in] el2 The second element attached to the face
   * @param[in] Tr The face element transformation
   * @param[in] elfun The current value of the underlying finite element state for residual evaluation
   * @param[out] elvec The evaluated residual
   */
  virtual void AssembleFaceVector(const mfem::FiniteElement& el1, const mfem::FiniteElement& el2,
                                  mfem::FaceElementTransformations& Tr, const mfem::Vector& elfun, mfem::Vector& elvec);

  /**
   * @brief Assemble the gradient for the nonlinear residual at a current state
   *
   * @param[in] el1 The first element attached to the face
   * @param[in] el2 The second element attached to the face
   * @param[in] Tr The face element transformation
   * @param[in] elfun The current value of the underlying finite element state for gradient evaluation
   * @param[out] elmat The local contribution to the Jacobian
   */
  virtual void AssembleFaceGrad(const mfem::FiniteElement& el1, const mfem::FiniteElement& el2,
                                mfem::FaceElementTransformations& Tr, const mfem::Vector& elfun,
                                mfem::DenseMatrix& elmat);

  /**
   * @brief Destroy the Hyperelastic Traction Integrator object
   */
  virtual ~HyperelasticTractionIntegrator() {}

private:
  /**
   * @brief The vector coefficient for the traction load
   */
  mfem::VectorCoefficient& function_;

  /**
   * @brief Working matrices for the traction calculations
   */
  mutable mfem::DenseMatrix DSh_u_;
  mutable mfem::DenseMatrix DS_u_;
  mutable mfem::DenseMatrix J0i_;
  mutable mfem::DenseMatrix F_;
  mutable mfem::DenseMatrix Finv_;
  mutable mfem::DenseMatrix FinvT_;
  mutable mfem::DenseMatrix PMatI_u_;

  /**
   * @brief Working vectors for the traction
   */
  mutable mfem::Vector shape_;
  mutable mfem::Vector nor_;
  mutable mfem::Vector fnor_;
  mutable mfem::Vector Sh_p_;
  mutable mfem::Vector Sh_u_;

  bool compute_on_reference_;
};

/**
 * @brief Custom MFEM integrator for pressure loads
 */
class HyperelasticPressureIntegrator : public mfem::NonlinearFormIntegrator {
public:
  /**
   * @brief Construct a new Hyperelastic Pressure Integrator object
   *
   * @param[in] p The pressure load coefficient
   * @param[in] compute_on_reference Flag for computing the residual in the stress-free configuration vs. the deformed
   * configuration
   */
  explicit HyperelasticPressureIntegrator(mfem::Coefficient& p, bool compute_on_reference = false)
      : pressure_(p), compute_on_reference_(compute_on_reference)
  {
  }

  /**
   * @brief Assemble the nonlinear residual on a boundary element at a current state
   *
   * @param[in] el1 The first element attached to the face
   * @param[in] el2 The second element attached to the face
   * @param[in] Tr The face element transformation
   * @param[in] elfun The current value of the underlying finite element state for residual evaluation
   * @param[out] elvec The evaluated residual
   */
  virtual void AssembleFaceVector(const mfem::FiniteElement& el1, const mfem::FiniteElement& el2,
                                  mfem::FaceElementTransformations& Tr, const mfem::Vector& elfun, mfem::Vector& elvec);

  /**
   * @brief Assemble the gradient for the nonlinear residual at a current state
   *
   * @param[in] el1 The first element attached to the face
   * @param[in] el2 The second element attached to the face
   * @param[in] Tr The face element transformation
   * @param[in] elfun The current value of the underlying finite element state for gradient evaluation
   * @param[out] elmat The local contribution to the Jacobian
   */
  virtual void AssembleFaceGrad(const mfem::FiniteElement& el1, const mfem::FiniteElement& el2,
                                mfem::FaceElementTransformations& Tr, const mfem::Vector& elfun,
                                mfem::DenseMatrix& elmat);

  /**
   * @brief Destroy the Hyperelastic Pressure Integrator object
   */
  virtual ~HyperelasticPressureIntegrator() {}

private:
  /**
   * @brief The coefficient for the pressure load
   */
  mfem::Coefficient& pressure_;

  /**
   * @brief Working matrices for the traction calculations
   */
  mutable mfem::DenseMatrix DSh_u_;
  mutable mfem::DenseMatrix DS_u_;
  mutable mfem::DenseMatrix J0i_;
  mutable mfem::DenseMatrix F_;
  mutable mfem::DenseMatrix Finv_;
  mutable mfem::DenseMatrix FinvT_;
  mutable mfem::DenseMatrix PMatI_u_;
  mutable mfem::DenseMatrix J_;
  mutable mfem::DenseMatrix Jinv_;

  /**
   * @brief Working vectors for the traction
   */
  mutable mfem::Vector shape_;
  mutable mfem::Vector nor_;
  mutable mfem::Vector fnor_;
  mutable mfem::Vector Sh_p_;
  mutable mfem::Vector Sh_u_;

  bool compute_on_reference_;
};

}  // namespace serac::mfem_ext

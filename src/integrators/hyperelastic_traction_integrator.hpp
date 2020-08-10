// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file hyperelastic_traction_integrator.hpp
 *
 * @brief Custom MFEM integrator for nonlinear finite deformation traction loads
 */

#include "mfem.hpp"

namespace serac {

/**
 * @brief Custom MFEM integrator for nonlinear finite deformation traction loads
 */
class HyperelasticTractionIntegrator : public mfem::NonlinearFormIntegrator {
public:
  /**
   * @brief Construct a new Hyperelastic Traction Integrator object
   *
   * @param[in] f The traction load coefficient
   */
  explicit HyperelasticTractionIntegrator(mfem::VectorCoefficient& f) : function_(f) {}

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
  mutable mfem::DenseMatrix DSh_u_, DS_u_, J0i_, F_, Finv_, FinvT_, PMatI_u_;

  /**
   * @brief Working vectors for the traction
   */
  mutable mfem::Vector shape_, nor_, fnor_, Sh_p_, Sh_u_;
};

}  // namespace serac

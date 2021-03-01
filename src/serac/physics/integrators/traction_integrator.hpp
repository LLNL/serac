// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file traction_integrator.hpp
 *
 * @brief Custom MFEM integrator for nonlinear finite deformation traction loads
 */

#pragma once

#include "mfem.hpp"

namespace serac::mfem_ext {

/**
 * @brief Custom MFEM integrator for nonlinear finite deformation traction loads
 */
class TractionIntegrator : public mfem::NonlinearFormIntegrator {
public:
  /**
   * @brief Construct a new Hyperelastic Traction Integrator object
   *
   * @param[in] f The traction load coefficient
   * @param[in] compute_on_reference Flag for computing the residual in the stress-free configuration vs. the deformed
   * configuration
   */
  explicit TractionIntegrator(mfem::VectorCoefficient& f, bool compute_on_reference = false)
      : traction_(f), compute_on_reference_(compute_on_reference)
  {
  }

  /**
   * @brief Assemble the nonlinear residual on a boundary facet at a current state
   *
   * @param[in] element_1 The first element attached to the face
   * @param[in] element_2 The second element attached to the face
   * @param[in] parent_to_reference_face_transformation The face element transformation
   * @param[in] input_state_vector The current value of the underlying finite element state for residual evaluation
   * @param[out] output_residual_vector The evaluated residual
   */
  virtual void AssembleFaceVector(const mfem::FiniteElement& element_1, const mfem::FiniteElement& element_2,
                                  mfem::FaceElementTransformations& parent_to_reference_face_transformation,
                                  const mfem::Vector& input_state_vector, mfem::Vector& output_residual_vector);

  /**
   * @brief Assemble the gradient for the nonlinear residual at a current state
   * @note When the traction is defined on the current configuration, this is computed via finite difference.
   *       This is not a performant method and is intended as a stop gap until an uncoming code refactoring.
   *
   * @param[in] element_1 The first element attached to the face
   * @param[in] element_2 The second element attached to the face
   * @param[in] parent_to_reference_face_transformation The face element transformation
   * @param[in] input_state_vector The current value of the underlying finite element state for gradient evaluation
   * @param[out] stiffness_matrix The local contribution to the Jacobian
   */
  virtual void AssembleFaceGrad(const mfem::FiniteElement& element_1, const mfem::FiniteElement& element_2,
                                mfem::FaceElementTransformations& parent_to_reference_face_transformation,
                                const mfem::Vector& input_state_vector, mfem::DenseMatrix& stiffness_matrix);

  /**
   * @brief Destroy the Hyperelastic Traction Integrator object
   */
  virtual ~TractionIntegrator() {}

private:
  /**
   * @brief The vector coefficient for the traction load
   */
  mfem::VectorCoefficient& traction_;

  /**
   * @brief gradients of shape functions on the parent element (dof x dim).
   *
   */
  mutable mfem::DenseMatrix dN_dxi_;

  /**
   * @brief gradients of the shape functions in the reference configuration (DN_i, DX_j)
   *
   */
  mutable mfem::DenseMatrix dN_dX_;

  /**
   * @brief the Jacobian of the reference to parent element transformation.
   *
   */
  mutable mfem::DenseMatrix dxi_dX_;

  /**
   * @brief the displacement gradient
   *
   */
  mutable mfem::DenseMatrix du_dX_;

  /**
   * @brief the deformation gradient
   *
   */
  mutable mfem::DenseMatrix F_;

  /**
   * @brief the inverse of the deformation gradient
   *
   */
  mutable mfem::DenseMatrix Finv_;

  /**
   * @brief Current input state dofs (dof x dim)
   *
   */
  mutable mfem::DenseMatrix input_state_matrix_;

  /**
   * @brief The basis functions
   *
   */
  mutable mfem::Vector shape_;

  /**
   * @brief The computed traction vector
   *
   */
  mutable mfem::Vector traction_vector_;

  /**
   * @brief The normal vector in the reference configuration
   *
   */
  mutable mfem::Vector reference_normal_;

  /**
   * @brief The normal vector in the deformed configuration
   *
   */
  mutable mfem::Vector current_normal_;

  /**
   * @brief Flag to compute on the reference configuration (linear traction)
   *
   */
  bool compute_on_reference_;
};

/**
 * @brief Custom MFEM integrator for pressure loads
 */
class PressureIntegrator : public mfem::NonlinearFormIntegrator {
public:
  /**
   * @brief Construct a new Hyperelastic Pressure Integrator object
   *
   * @param[in] p The pressure load coefficient
   * @param[in] compute_on_reference Flag for computing the residual in the stress-free configuration vs. the deformed
   * configuration
   */
  explicit PressureIntegrator(mfem::Coefficient& p, bool compute_on_reference = false)
      : pressure_(p), compute_on_reference_(compute_on_reference)
  {
  }

  /**
   * @brief Assemble the nonlinear residual on a boundary element at a current state
   *
   * @param[in] element_1 The first element attached to the face
   * @param[in] element_2 The second element attached to the face
   * @param[in] parent_to_reference_face_transformation The face element transformation
   * @param[in] input_state_vector The current value of the underlying finite element state for residual evaluation
   * @param[out] output_residual_vector The evaluated residual
   */
  virtual void AssembleFaceVector(const mfem::FiniteElement& element_1, const mfem::FiniteElement& element_2,
                                  mfem::FaceElementTransformations& parent_to_reference_face_transformation,
                                  const mfem::Vector& input_state_vector, mfem::Vector& output_residual_vector);

  /**
   * @brief Assemble the gradient for the nonlinear residual at a current state
   * @note When the traction is defined on the current configuration, this is computed via finite difference.
   *       This is not a performant method and is intended as a stop gap until an uncoming code refactoring.
   *
   * @param[in] element_1 The first element attached to the face
   * @param[in] element_2 The second element attached to the face
   * @param[in] parent_to_reference_face_transformation The face element transformation
   * @param[in] input_state_vector The current value of the underlying finite element state for gradient evaluation
   * @param[out] stiffness_matrix The local contribution to the Jacobian
   */
  virtual void AssembleFaceGrad(const mfem::FiniteElement& element_1, const mfem::FiniteElement& element_2,
                                mfem::FaceElementTransformations& parent_to_reference_face_transformation,
                                const mfem::Vector& input_state_vector, mfem::DenseMatrix& stiffness_matrix);

  /**
   * @brief Destroy the Hyperelastic Pressure Integrator object
   */
  virtual ~PressureIntegrator() {}

private:
  /**
   * @brief The coefficient for the pressure load
   */
  mfem::Coefficient& pressure_;

  /**
   * @brief gradients of shape functions on the parent element (dof x dim).
   *
   */
  mutable mfem::DenseMatrix dN_dxi_;

  /**
   * @brief gradients of the shape functions in the reference configuration (DN_i, DX_j)
   *
   */
  mutable mfem::DenseMatrix dN_dX_;

  /**
   * @brief the Jacobian of the reference to parent element transformation.
   *
   */
  mutable mfem::DenseMatrix dxi_dX_;

  /**
   * @brief the displacement gradient
   *
   */
  mutable mfem::DenseMatrix du_dX_;

  /**
   * @brief the deformation gradient
   *
   */
  mutable mfem::DenseMatrix F_;

  /**
   * @brief the inverse of the deformation gradient
   *
   */
  mutable mfem::DenseMatrix Finv_;

  /**
   * @brief Current input state dofs (dof x dim)
   *
   */
  mutable mfem::DenseMatrix input_state_matrix_;

  /**
   * @brief The basis functions
   *
   */
  mutable mfem::Vector shape_;

  /**
   * @brief The normal vector in the reference configuration
   *
   */
  mutable mfem::Vector reference_normal_;

  /**
   * @brief The normal vector in the deformed configuration
   *
   */
  mutable mfem::Vector current_normal_;

  /**
   * @brief Flag to compute on the reference configuration (linear traction)
   *
   */
  bool compute_on_reference_;
};

}  // namespace serac::mfem_ext

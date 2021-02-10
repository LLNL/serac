// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file displacement_hyperelastic_integrator.hpp
 *
 * @brief The MFEM integrators for the displacement hyperelastic formulation
 */

#pragma once

#include "serac/physics/materials/hyperelastic_material.hpp"

#include "mfem.hpp"

namespace serac::mfem_ext {

/**
 * @brief Displacement hyperelastic integrator for any given serac::HyperelasticModel.
 *
 */
class DisplacementHyperelasticIntegrator : public mfem::NonlinearFormIntegrator {
public:
  /**
   * @brief The constructor for the displacement hyperelastic integrator
   *
   * @param[in] m  HyperelasticModel that will be integrated.
   * @param[in] geom_nonlin Flag to include geometric nonlinearities in the residual calculation
   */
  explicit DisplacementHyperelasticIntegrator(HyperelasticMaterial& m, bool geom_nonlin = true)
      : material_(m), geom_nonlin_(geom_nonlin)
  {
  }

  /**
   * @brief Computes the integral of W(Jacobian(Trt)) over a target zone
   *
   * @param[in] element     Type of FiniteElement.
   * @param[in] basis_to_reference_transformation    Represents ref->target coordinates transformation.
   * @param[in] state_vector  Physical coordinates of the zone.
   */
  virtual double GetElementEnergy(const mfem::FiniteElement&   element,
                                  mfem::ElementTransformation& basis_to_reference_transformation,
                                  const mfem::Vector&          state_vector);

  /**
   * @brief The residual evaluation for the nonlinear incremental integrator
   *
   * @param[in] element The finite element to integrate
   * @param[in] basis_to_reference_transformation The element transformation operators
   * @param[in] state_vector The state vector to evaluate the residual
   * @param[out] residual The output residual
   */
  virtual void AssembleElementVector(const mfem::FiniteElement&   element,
                                     mfem::ElementTransformation& basis_to_reference_transformation,
                                     const mfem::Vector& state_vector, mfem::Vector& residual);

  /**
   * @brief Assemble the local gradient
   *
   * @param[in] element The finite element to integrate
   * @param[in] basis_to_reference_transformation The element transformation operators
   * @param[in] state_vector The state vector to evaluate the gradient
   * @param[out] elmat The output local gradient
   */
  virtual void AssembleElementGrad(const mfem::FiniteElement&   element,
                                   mfem::ElementTransformation& basis_to_reference_transformation,
                                   const mfem::Vector& state_vector, mfem::DenseMatrix& stiffness_matrix);

private:
  /**
   * @brief Calculate the deformation gradient and right Cauchy-Green deformation tensor at a quadrature point
   *
   * @param[in] element The finite element
   * @param[in] int_point The integration point
   * @param[in] basis_to_reference_transformation The reference-to-target (stress-free) transformation
   */
  void CalcKinematics(const mfem::FiniteElement& element, const mfem::IntegrationPoint& int_point,
                      mfem::ElementTransformation& basis_to_reference_transformation);

  /**
   * @brief The associated hyperelastic model
   */
  HyperelasticMaterial& material_;

  /**
   * @brief gradients of shape functions on the parent element (dof x dim).
   *
   */
  mfem::DenseMatrix dN_dxi_;

  /**
   * @brief gradients of the shape functions in the reference configuration (DN_i, DX_j)
   *
   */
  mfem::DenseMatrix dN_dX_;

  /**
   * @brief gradients of the shape functions in the current configuration (DN_i, Dx_j)
   *
   */
  mfem::DenseMatrix B_;

  /**
   * @brief the Jacobian of the reference to parent element transformation.
   *
   */
  mfem::DenseMatrix dxi_dX_;

  /**
   * @brief the deformation gradient
   *
   */
  mfem::DenseMatrix F_;

  /**
   * @brief the inverse of the deformation gradient
   *
   */
  mfem::DenseMatrix Finv_;

  /**
   * @brief the spatial tangent moduli
   *
   */
  serac::mfem_ext::Array4D<double> C_;

  /**
   * @brief the Cauchy stress
   *
   */
  mfem::DenseMatrix sigma_;

  /**
   * @brief Current input state dofs (dof x dim)
   *
   */
  mfem::DenseMatrix input_state_matrix_;

  /**
   * @brief Current residual contribution (dof x dim)
   *
   */
  mfem::DenseMatrix output_residual_matrix_;

  /**
   * @brief The displacement gradient
   *
   */
  mfem::DenseMatrix du_dX_;

  /**
   * @brief The determinant of the deformation gradient
   *
   */
  double det_J_;

  /**
   * @brief The geometric nonlinearity flag
   */
  bool geom_nonlin_;
};

}  // namespace serac::mfem_ext

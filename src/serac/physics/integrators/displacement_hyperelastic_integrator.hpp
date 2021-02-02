// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file inc_hyperelastic_integrator.hpp
 *
 * @brief The MFEM integrators for the incremental hyperelastic formulation
 */

#ifndef DISPLACEMENT_INTEGRATOR_HPP
#define DISPLACEMENT_INTEGRATOR_HPP

#include "serac/physics/materials/hyperelastic_material.hpp"

#include "mfem.hpp"

namespace serac {

/**
 * @brief Incremental hyperelastic integrator for any given HyperelasticModel.
 *
 */
class DisplacementHyperelasticIntegrator : public mfem::NonlinearFormIntegrator {
public:
  /**
   * @brief The constructor for the incremental hyperelastic integrator
   *
   * @param[in] m  HyperelasticModel that will be integrated.
   */
  explicit DisplacementHyperelasticIntegrator(serac::HyperelasticMaterial& m, bool geom_nonlin = true)
      : material_(m), geom_nonlin_(geom_nonlin)
  { }

  /**
   * @brief Computes the integral of W(Jacobian(Trt)) over a target zone
   *
   * @param[in] el     Type of FiniteElement.
   * @param[in] Ttr    Represents ref->target coordinates transformation.
   * @param[in] elfun  Physical coordinates of the zone.
   */
  virtual double GetElementEnergy(const mfem::FiniteElement& el, mfem::ElementTransformation& Ttr,
                                  const mfem::Vector& elfun);

  /**
   * @brief The residual evaluation for the nonlinear incremental integrator
   *
   * @param[in] el The finite element to integrate
   * @param[in] Ttr The element transformation operators
   * @param[in] elfun The state vector to evaluate the residual
   * @param[out] elvect The output residual
   */
  virtual void AssembleElementVector(const mfem::FiniteElement& el, mfem::ElementTransformation& Ttr,
                                     const mfem::Vector& elfun, mfem::Vector& elvect);

  /**
   * @brief Assemble the local gradient
   *
   * @param[in] el The finite element to integrate
   * @param[in] Ttr The element transformation operators
   * @param[in] elfun The state vector to evaluate the gradient
   * @param[out] elmat The output local gradient
   */
  virtual void AssembleElementGrad(const mfem::FiniteElement& el, mfem::ElementTransformation& Ttr,
                                   const mfem::Vector& elfun, mfem::DenseMatrix& elmat);

private:
  /**
   * @brief Calculate the deformation gradient and right Cauchy-Green deformation tensor at a quadrature point
   *
   * @param[in] el The finite element
   * @param[in] ip The integration point
   * @param[in] Ttr The reference-to-target (stress-free) transformation
   */
  void CalcDeformationGradient(const mfem::FiniteElement& el, const mfem::IntegrationPoint& ip,
                               mfem::ElementTransformation& Ttr);

  /**
   * @brief The associated hyperelastic model
   */
  serac::HyperelasticMaterial& material_;

  /**
   * @brief gradients of reference shape functions (dof x dim).
   *
   */
  mfem::DenseMatrix DSh_;

  /**
   * @brief gradients of the shape functions in the target configuration (DN_i, DX_j)
   *
   */
  mfem::DenseMatrix DS_;

  /**
   * @brief gradients of the shape functions in the current configuration (DN_i, Dx_j)
   *
   */
  mfem::DenseMatrix B_;

  /**
   * @brief the Jacobian of the target-to-reference-element transformation.
   *
   */
  mfem::DenseMatrix Jrt_;

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
  mfem_ext::Array4D<double> C_;

  /**
   * @brief the Cauchy stress
   *
   */
  mfem::DenseMatrix sigma_;

  /**
   * @brief Current input state dofs (dof x dim)
   *
   */
  mfem::DenseMatrix PMatI_;

  /**
   * @brief Current residual contribution (dof x dim)
   *
   */
  mfem::DenseMatrix PMatO_;

  /**
   * @brief Assembled material stiffness contributions
   *
   */
  mfem::DenseMatrix K_;

  /**
   * @brief The displacement gradient
   *
   */
  mfem::DenseMatrix H_;

  /**
   * @brief The determinant of the deformation gradient
   *
   */
  double J_;

  /**
   * @brief The geometric nonlinearity flag
   */
  bool geom_nonlin_;
};

}  // namespace serac

#endif

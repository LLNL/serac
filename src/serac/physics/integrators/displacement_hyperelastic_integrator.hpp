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
  explicit DisplacementHyperelasticIntegrator(serac::HyperelasticMaterial& m, const int dim, bool geom_nonlin = true)
      : material_(m), geom_nonlin_(geom_nonlin)
  {
    getShearTerms(dim, shear_terms_);
    eye_.SetSize(dim);
    eye_ = 0.0;
    for (int i = 0; i < dim; ++i) {
      eye_(i, i) = 1.0;
    }
  }

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
   * @brief Calculate the B matrix in Voigt notation
   *
   */
  void CalcBMatrix();

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
   * @brief right Cauchy-Green deformation tensor
   *
   */
  mfem::DenseMatrix C_;

  /**
   * @brief Tangent stiffness module in Voigt notation
   *
   */
  mfem::DenseMatrix T_;

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
   * @brief Temporary matrix for calculating matrix-matrix products
   *
   */
  mfem::DenseMatrix temp_;

  /**
   * @brief Assembled material stiffness contributions
   *
   */
  mfem::DenseMatrix K_;

  /**
   * @brief B_0T matrix (in Voigt notation) for a specific shape function
   *
   */
  mfem::DenseMatrix B_0T;

  /**
   * @brief The displacement gradient
   *
   */
  mfem::DenseMatrix H_;

  /**
   * @brief The transpose of the displcement gradient
   *
   */
  mfem::DenseMatrix HT_;

  /**
   * @brief The identity matrix
   *
   */
  mfem::DenseMatrix eye_;

  /**
   * @brief The PK2 stress in tensor form
   *
   */
  mfem::DenseMatrix S_mat_;

  /**
   * @brief The vector of shear terms for Voigt notation
   *
   */
  std::vector<std::pair<int, int>> shear_terms_;

  /**
   * @brief A vector of the Voigt-notation B matrix for each shape function
   *
   */
  std::vector<mfem::DenseMatrix> B_0_;

  /**
   * @brief The PK2 stress in Voigt notation
   *
   */
  mfem::Vector S_;

  /**
   * @brief The local contributions to the residual vector
   *
   */
  mfem::Vector force_;

  /**
   * @brief The geometric nonlinearity flag
   */
  bool geom_nonlin_;
};

}  // namespace serac

#endif

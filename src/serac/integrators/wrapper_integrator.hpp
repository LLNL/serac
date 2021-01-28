// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file wrapper_integrator.hpp
 *
 * @brief Wrappers to turn bilinear and linear integrators into nonlinear ones
 */

#pragma once

#include <functional>
#include <memory>

#include "mfem.hpp"

namespace serac::mfem_ext {

/**
 *  @brief A class to convert linearform integrators into a nonlinear residual-based one
 */
class LinearToNonlinearFormIntegrator : public mfem::NonlinearFormIntegrator {
public:
  /**
   * @brief Recasts, A(u) = F as R(u) = A(u) - F
   *
   * @param[in] f A LinearFormIntegrator
   * @param[in] trial_fes The trial finite element space
   */
  explicit LinearToNonlinearFormIntegrator(std::shared_ptr<mfem::LinearFormIntegrator>  f,
                                           std::shared_ptr<mfem::ParFiniteElementSpace> trial_fes);

  /**
   * @brief Compute the residual vector => -F
   *
   * @param[in] el The finite element for local integration
   * @param[in] Tr The local FE transformation
   * @param[in] elfun The state to evaluate the residual
   * @param[out] elvect The output residual
   */
  virtual void AssembleElementVector(const mfem::FiniteElement& el, mfem::ElementTransformation& Tr,
                                     const mfem::Vector& elfun, mfem::Vector& elvect);

  /**
   * @brief Compute the tangent matrix = 0
   *
   * @param[in] el The finite element for local integration
   * @param[in] Tr The local FE transformation
   * @param[in] elfun The state to evaluate the residual
   * @param[out] elmat elvect The output gradient
   */
  virtual void AssembleElementGrad(const mfem::FiniteElement& el, mfem::ElementTransformation& Tr,
                                   const mfem::Vector& elfun, mfem::DenseMatrix& elmat);

private:
  /**
   * @brief The linear form integrator to wrap
   */
  std::shared_ptr<mfem::LinearFormIntegrator> f_;

  /**
   * @brief The trial FE space
   */
  std::shared_ptr<mfem::ParFiniteElementSpace> trial_fes_;
};

/**
 * @brief A class to convert bilinearform integrators into a nonlinear residual-based one
 */
class BilinearToNonlinearFormIntegrator : public mfem::NonlinearFormIntegrator {
public:
  /**
   * @brief Recasts, A(u) = F as R(u) = A(u) - F
   *
   * @param[in] A A BilinearFormIntegrator
   */
  explicit BilinearToNonlinearFormIntegrator(std::shared_ptr<mfem::BilinearFormIntegrator> A);

  /**
   * @brief Compute the residual vector
   * @param[in] el The finite element for local integration
   * @param[in] Tr The local FE transformation
   * @param[in] elfun The state to evaluate the residual
   * @param[out] elvect The output residual
   */
  virtual void AssembleElementVector(const mfem::FiniteElement& el, mfem::ElementTransformation& Tr,
                                     const mfem::Vector& elfun, mfem::Vector& elvect);

  /**
   * @brief Compute the tangent matrix = 0
   *
   * @param[in] el The finite element for local integration
   * @param[in] Tr The local FE transformation
   * @param[in] elfun The state to evaluate the residual
   * @param[out] elmat elvect The output gradient
   */
  virtual void AssembleElementGrad(const mfem::FiniteElement& el, mfem::ElementTransformation& Tr,
                                   const mfem::Vector& elfun, mfem::DenseMatrix& elmat);

private:
  /**
   * @brief The bilinear form to wrap
   *
   */
  std::shared_ptr<mfem::BilinearFormIntegrator> A_;
};

/**
 * @brief A class to convert a MixedBiolinearIntegrator into a nonlinear residual-based one
 */
class MixedBilinearToNonlinearFormIntegrator : public mfem::NonlinearFormIntegrator {
public:
  /**
   * @brief Recasts, A(u) = F as R(u) = A(u) - F
   *
   * @param[in] A A MixedBilinearFormIntegrator
   * @param[in] trial_fes The trial finite element space
   */
  MixedBilinearToNonlinearFormIntegrator(std::shared_ptr<mfem::BilinearFormIntegrator> A,
                                         std::shared_ptr<mfem::ParFiniteElementSpace>  trial_fes);

  /**
   * @brief Compute the residual vector => -F
   *
   * @param[in] el The finite element for local integration
   * @param[in] Tr The local FE transformation
   * @param[in] elfun The state to evaluate the residual
   * @param[out] elvect The output residual
   */
  virtual void AssembleElementVector(const mfem::FiniteElement& el, mfem::ElementTransformation& Tr,
                                     const mfem::Vector& elfun, mfem::Vector& elvect);

  /**
   * @brief Compute the tangent matrix = 0
   *
   * @param[in] el The finite element for local integration
   * @param[in] Tr The local FE transformation
   * @param[in] elfun The state to evaluate the residual
   * @param[out] elmat elvect The output gradient
   */
  virtual void AssembleElementGrad(const mfem::FiniteElement& el, mfem::ElementTransformation& Tr,
                                   const mfem::Vector& elfun, mfem::DenseMatrix& elmat);

private:
  /**
   * @brief The bilinear integrator to wrap
   */
  std::shared_ptr<mfem::BilinearFormIntegrator> A_;

  /**
   * @brief The trial finite element space
   */
  std::shared_ptr<mfem::ParFiniteElementSpace> trial_fes_;
};

/**
 * @brief A class to convert NonlinearFormIntegrator to one where the input parameter undergoes a change of variables
 */
class TransformedNonlinearFormIntegrator : public mfem::NonlinearFormIntegrator {
public:
  /// alias for prototype of the residual_func
  using transformed_func = mfem::Vector(const mfem::FiniteElement& el, mfem::ElementTransformation& Tr,
                                        const mfem::Vector&);

  /// alias for prototype of the gradient of residual_func
  using transformed_gradient_func = mfem::DenseMatrix(const mfem::FiniteElement& el, mfem::ElementTransformation& Tr,
                                                      const mfem::DenseMatrix&);

  /**
   * @brief Recasts, A(u(x)) = F as R(u(x)) = A(u(x)) - F = R(x)
   *
   * @param[in] R A BilinearFormIntegrator
   * @param[in] transformed_input A function that performs a change of variables to what R expects
   * @param[in] transformed_grad_output A function that performs a change of variables for the gradient
   */
  explicit TransformedNonlinearFormIntegrator(std::shared_ptr<mfem::NonlinearFormIntegrator> R,
                                              std::function<transformed_func>                transformed_input,
                                              std::function<transformed_gradient_func>       transformed_grad_output);

  /**
   * @brief Compute the residual vector with input, transformed_function(x)
   * @param[in] el The finite element for local integration
   * @param[in] Tr The local FE transformation
   * @param[in] elfun The state to evaluate the residual
   * @param[out] elvect The output residual
   */
  virtual void AssembleElementVector(const mfem::FiniteElement& el, mfem::ElementTransformation& Tr,
                                     const mfem::Vector& elfun, mfem::Vector& elvect);

  /**
   * @brief Compute the tangent matrix with input, transformed_function(x)
   *
   * @param[in] el The finite element for local integration
   * @param[in] Tr The local FE transformation
   * @param[in] elfun The state to evaluate the residual
   * @param[out] elmat elvect The output gradient
   */
  virtual void AssembleElementGrad(const mfem::FiniteElement& el, mfem::ElementTransformation& Tr,
                                   const mfem::Vector& elfun, mfem::DenseMatrix& elmat);

private:
  /**
   * @brief The NonlinearFormIntegrator form to wrap
   *
   */
  std::shared_ptr<mfem::NonlinearFormIntegrator> R_;

  /**
   * @brief The transforming function on input x
   *
   */

  std::function<transformed_func> transformed_function_;

  /**
   * @brief The transforming function to perform change of variables for the gradient
   */
  std::function<transformed_gradient_func> transformed_function_grad_;
};

}  // namespace serac::mfem_ext

// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file wrapper_integrator.hpp
 *
 * @brief Wrappers to turn bilinear and linear integrators into nonlinear ones
 */

#ifndef WRAPPER_INTEGRATOR_HPP
#define WRAPPER_INTEGRATOR_HPP

#include <memory>
#include <functional>

#include "mfem.hpp"

namespace serac {

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
 * @brief A class to convert NonlinearFormIntegrator to one where the input parameter undergoes variable substitution
 */
class SubstitutionNonlinearFormIntegrator : public mfem::NonlinearFormIntegrator {
public:
  /**
   * @brief Recasts, A(u(x)) = F as R(u(x)) = A(u(x)) - F = R(x)
   *
   * @param[in] R A BilinearFormIntegrator
   */
  explicit SubstitutionNonlinearFormIntegrator(std::shared_ptr<mfem::NonlinearFormIntegrator> R,
					       std::function<std::shared_ptr<mfem::Vector>(const mfem::Vector &)> substitute,
					       std::function<std::shared_ptr<mfem::DenseMatrix>(const mfem::DenseMatrix &)> substitute_back);

  /**
   * @brief Compute the residual vector with input, substitute_function(x)
   * @param[in] el The finite element for local integration
   * @param[in] Tr The local FE transformation
   * @param[in] elfun The state to evaluate the residual
   * @param[out] elvect The output residual
   */
  virtual void AssembleElementVector(const mfem::FiniteElement& el, mfem::ElementTransformation& Tr,
                                     const mfem::Vector& elfun, mfem::Vector& elvect);

  /**
   * @brief Compute the tangent matrix with input, substitute_function(x)
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
   * @brief The substitution function on input x
   *
   */

  std::function<std::shared_ptr<mfem::Vector>(const mfem::Vector &)> substitute_function_;

  /**
   * @brief The substitution to perform change of variables for the tangent back to x
   */
  std::function<std::shared_ptr<mfem::DenseMatrix>(const mfem::DenseMatrix &)> substitute_function_back_;

};

  
}  // namespace serac

#endif

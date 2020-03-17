// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "mfem.hpp"

/// A class to convert linearform integrators into a nonlinear residual-based one
class LinearNonlinearFormIntegrator : public mfem::NonlinearFormIntegrator
{
public:
  /**
     \brief Recasts, A(u) = F as R(u) = A(u) - F
     
\param[in] f A LinearFormIntegrator
   */
  LinearNonlinearFormIntegrator( mfem::LinearFormIntegrator & f, mfem::ParFiniteElementSpace *trial_fes);

  /// Compute the residual vector => -F
  virtual void AssembleElementVector( const mfem::FiniteElement &el, mfem::ElementTransformation &Tr, const mfem::Vector &elfun, mfem::Vector & elvect);

  /// Compute the tangent matrix = 0
  virtual void AssembleElementGrad( const mfem::FiniteElement &el, mfem::ElementTransformation &Tr, const mfem::Vector &elfun, mfem::DenseMatrix &elmat);

  virtual ~LinearNonlinearFormIntegrator();
  
private:
  mfem::LinearFormIntegrator & m_f;
  mfem::ParFiniteElementSpace *m_trial_fes;
};


/// A class to convert linearform integrators into a nonlinear residual-based one
class BilinearNonlinearFormIntegrator : public mfem::NonlinearFormIntegrator
{
public:
  /**
     \brief Recasts, A(u) = F as R(u) = A(u) - F

     \param[in] A A BilinearFormIntegrator
   */
  BilinearNonlinearFormIntegrator( mfem::BilinearFormIntegrator & A);

  /// Compute the residual vector
  virtual void AssembleElementVector( const mfem::FiniteElement &el, mfem::ElementTransformation &Tr, const mfem::Vector &elfun, mfem::Vector & elvect);

  /// Compute the tangent matrix
  virtual void AssembleElementGrad( const mfem::FiniteElement &el, mfem::ElementTransformation &Tr, const mfem::Vector &elfun, mfem::DenseMatrix &elmat);

  virtual ~BilinearNonlinearFormIntegrator();
  
private:
  mfem::BilinearFormIntegrator & m_A;
  mfem::ParFiniteElementSpace *m_trial_fes;
};

/// A class to convert a MixedBiolinearIntegrator into a nonlinear residual-based one
class MixedBilinearNonlinearFormIntegrator : public mfem::NonlinearFormIntegrator
{
public:
  /**
     \brief Recasts, A(u) = F as R(u) = A(u) - F

     \param[in] A A MixedBilinearFormIntegrator
   */
  MixedBilinearNonlinearFormIntegrator( mfem::BilinearFormIntegrator & A, mfem::ParFiniteElementSpace *trial_fes);

  /// Compute the residual vector
  virtual void AssembleElementVector( const mfem::FiniteElement &el, mfem::ElementTransformation &Tr, const mfem::Vector &elfun, mfem::Vector & elvect);

  /// Compute the tangent matrix
  virtual void AssembleElementGrad( const mfem::FiniteElement &el, mfem::ElementTransformation &Tr, const mfem::Vector &elfun, mfem::DenseMatrix &elmat);

  virtual ~MixedBilinearNonlinearFormIntegrator();
  
private:
  mfem::BilinearFormIntegrator & m_A;
  mfem::ParFiniteElementSpace *m_trial_fes;
};


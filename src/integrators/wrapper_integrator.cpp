// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

# include "wrapper_integrator.hpp"

LinearToNonlinearFormIntegrator::LinearToNonlinearFormIntegrator( mfem::LinearFormIntegrator & f,
    mfem::ParFiniteElementSpace * trial_fes)
  :
  m_f(f), m_trial_fes(trial_fes)
{}

void LinearToNonlinearFormIntegrator::AssembleElementVector( const mfem::FiniteElement &el,
    mfem::ElementTransformation &Tr,
    const mfem::Vector &elfun, mfem::Vector & elvect)
{
  m_f.AssembleRHSElementVect(el, Tr, elvect);
  elvect *= -1.;
}

void LinearToNonlinearFormIntegrator::AssembleElementGrad( const mfem::FiniteElement &el, mfem::ElementTransformation &Tr,
    const mfem::Vector &elfun, mfem::DenseMatrix &elmat)
{
  const mfem::FiniteElement &trial_el = *(m_trial_fes->FEColl()->FiniteElementForGeometry(Tr.GetGeometryType()));

  elmat.SetSize(trial_el.GetDof()*trial_el.GetDim(),
                el.GetDof()*el.GetDim());
  elmat = 0.;
}

BilinearToNonlinearFormIntegrator::BilinearToNonlinearFormIntegrator( mfem::BilinearFormIntegrator &A)
  :
  m_A(A)
{}

void BilinearToNonlinearFormIntegrator::AssembleElementVector( const mfem::FiniteElement &el,
    mfem::ElementTransformation &Tr,
    const mfem::Vector &elfun, mfem::Vector & elvect)
{
  mfem::DenseMatrix elmat;
  m_A.AssembleElementMatrix(el, Tr, elmat);
  elvect.SetSize(elmat.Height());
  elmat.Mult(elfun, elvect);
}

void BilinearToNonlinearFormIntegrator::AssembleElementGrad( const mfem::FiniteElement &el,
    mfem::ElementTransformation &Tr,
    const mfem::Vector &elfun, mfem::DenseMatrix &elmat)
{
  m_A.AssembleElementMatrix(el, Tr, elmat);
}

MixedBilinearToNonlinearFormIntegrator::MixedBilinearToNonlinearFormIntegrator( mfem::BilinearFormIntegrator &A,
    mfem::ParFiniteElementSpace *trial_fes)
  :
  m_A(A), m_trial_fes(trial_fes)
{}

void MixedBilinearToNonlinearFormIntegrator::AssembleElementVector( const mfem::FiniteElement &el,
    mfem::ElementTransformation &Tr,
    const mfem::Vector &elfun, mfem::Vector & elvect)
{
  const mfem::FiniteElement &trial_el = *(m_trial_fes->FEColl()->FiniteElementForGeometry(Tr.GetGeometryType()));

  mfem::DenseMatrix elmat;
  m_A.AssembleElementMatrix2(trial_el, el, Tr, elmat);
  elvect.SetSize(elmat.Height());
  elmat.Mult(elfun, elvect);
}

void MixedBilinearToNonlinearFormIntegrator::AssembleElementGrad( const mfem::FiniteElement &el,
    mfem::ElementTransformation &Tr,
    const mfem::Vector &elfun, mfem::DenseMatrix &elmat)
{
  const mfem::FiniteElement &trial_el = *(m_trial_fes->FEColl()->FiniteElementForGeometry(Tr.GetGeometryType()));

  m_A.AssembleElementMatrix2(trial_el, el, Tr, elmat);
}

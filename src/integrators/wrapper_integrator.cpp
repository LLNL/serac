// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

# include "wrapper_integrator.hpp"

LinearNonlinearFormIntegrator::LinearNonlinearFormIntegrator( mfem::LinearFormIntegrator & f,
    mfem::ParFiniteElementSpace * trial_fes)
  :
  m_f(f), m_trial_fes(trial_fes)
{}

void LinearNonlinearFormIntegrator::AssembleElementVector( const mfem::FiniteElement &el,
    mfem::ElementTransformation &Tr,
    const mfem::Vector &elfun, mfem::Vector & elvect)
{
  m_f.AssembleRHSElementVect(el, Tr, elvect);
  elvect *= -1.;
}

void LinearNonlinearFormIntegrator::AssembleElementGrad( const mfem::FiniteElement &el, mfem::ElementTransformation &Tr,
    const mfem::Vector &elfun, mfem::DenseMatrix &elmat)
{
  const mfem::FiniteElement &trial_el = *(m_trial_fes->FEColl()->FiniteElementForGeometry(Tr.GetGeometryType()));

  elmat.SetSize(trial_el.GetDof()*trial_el.GetDim(),
                el.GetDof()*el.GetDim());
  elmat = 0.;
}

BilinearNonlinearFormIntegrator::BilinearNonlinearFormIntegrator( mfem::BilinearFormIntegrator &A)
  :
  m_A(A)
{}

void BilinearNonlinearFormIntegrator::AssembleElementVector( const mfem::FiniteElement &el,
    mfem::ElementTransformation &Tr,
    const mfem::Vector &elfun, mfem::Vector & elvect)
{
  mfem::DenseMatrix elmat;
  m_A.AssembleElementMatrix(el, Tr, elmat);
  elvect.SetSize(elmat.Height());
  elmat.Mult(elfun, elvect);
}

void BilinearNonlinearFormIntegrator::AssembleElementGrad( const mfem::FiniteElement &el,
    mfem::ElementTransformation &Tr,
    const mfem::Vector &elfun, mfem::DenseMatrix &elmat)
{
  m_A.AssembleElementMatrix(el, Tr, elmat);
}

MixedBilinearNonlinearFormIntegrator::MixedBilinearNonlinearFormIntegrator( mfem::BilinearFormIntegrator &A,
    mfem::ParFiniteElementSpace *trial_fes)
  :
  m_A(A), m_trial_fes(trial_fes)
{}

void MixedBilinearNonlinearFormIntegrator::AssembleElementVector( const mfem::FiniteElement &el,
    mfem::ElementTransformation &Tr,
    const mfem::Vector &elfun, mfem::Vector & elvect)
{
  const mfem::FiniteElement &trial_el = *(m_trial_fes->FEColl()->FiniteElementForGeometry(Tr.GetGeometryType()));

  mfem::DenseMatrix elmat;
  m_A.AssembleElementMatrix2(trial_el, el, Tr, elmat);
  elvect.SetSize(elmat.Height());
  elmat.Mult(elfun, elvect);
}

void MixedBilinearNonlinearFormIntegrator::AssembleElementGrad( const mfem::FiniteElement &el,
    mfem::ElementTransformation &Tr,
    const mfem::Vector &elfun, mfem::DenseMatrix &elmat)
{
  const mfem::FiniteElement &trial_el = *(m_trial_fes->FEColl()->FiniteElementForGeometry(Tr.GetGeometryType()));

  m_A.AssembleElementMatrix2(trial_el, el, Tr, elmat);
}

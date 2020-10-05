// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "integrators/wrapper_integrator.hpp"

namespace serac {

LinearToNonlinearFormIntegrator::LinearToNonlinearFormIntegrator(std::shared_ptr<mfem::LinearFormIntegrator>  f,
                                                                 std::shared_ptr<mfem::ParFiniteElementSpace> trial_fes)
    : f_(f), trial_fes_(trial_fes)
{
}

void LinearToNonlinearFormIntegrator::AssembleElementVector(const mfem::FiniteElement&   el,
                                                            mfem::ElementTransformation& Tr, const mfem::Vector&,
                                                            mfem::Vector&                elvect)
{
  f_->AssembleRHSElementVect(el, Tr, elvect);
  elvect *= -1.;
}

void LinearToNonlinearFormIntegrator::AssembleElementGrad(const mfem::FiniteElement&   el,
                                                          mfem::ElementTransformation& Tr, const mfem::Vector&,
                                                          mfem::DenseMatrix&           elmat)
{
  const mfem::FiniteElement& trial_el = *(trial_fes_->FEColl()->FiniteElementForGeometry(Tr.GetGeometryType()));

  elmat.SetSize(trial_el.GetDof() * trial_el.GetDim(), el.GetDof() * el.GetDim());
  elmat = 0.;
}

BilinearToNonlinearFormIntegrator::BilinearToNonlinearFormIntegrator(std::shared_ptr<mfem::BilinearFormIntegrator> A)
    : A_(A)
{
}

void BilinearToNonlinearFormIntegrator::AssembleElementVector(const mfem::FiniteElement&   el,
                                                              mfem::ElementTransformation& Tr,
                                                              const mfem::Vector& elfun, mfem::Vector& elvect)
{
  mfem::DenseMatrix elmat;
  A_->AssembleElementMatrix(el, Tr, elmat);
  elvect.SetSize(elmat.Height());
  elmat.Mult(elfun, elvect);
}

void BilinearToNonlinearFormIntegrator::AssembleElementGrad(const mfem::FiniteElement&   el,
                                                            mfem::ElementTransformation& Tr, const mfem::Vector&,
                                                            mfem::DenseMatrix&           elmat)
{
  A_->AssembleElementMatrix(el, Tr, elmat);
}

MixedBilinearToNonlinearFormIntegrator::MixedBilinearToNonlinearFormIntegrator(
    std::shared_ptr<mfem::BilinearFormIntegrator> A, std::shared_ptr<mfem::ParFiniteElementSpace> trial_fes)
    : A_(A), trial_fes_(trial_fes)
{
}

void MixedBilinearToNonlinearFormIntegrator::AssembleElementVector(const mfem::FiniteElement&   el,
                                                                   mfem::ElementTransformation& Tr,
                                                                   const mfem::Vector& elfun, mfem::Vector& elvect)
{
  const mfem::FiniteElement& trial_el = *(trial_fes_->FEColl()->FiniteElementForGeometry(Tr.GetGeometryType()));

  mfem::DenseMatrix elmat;
  A_->AssembleElementMatrix2(trial_el, el, Tr, elmat);
  elvect.SetSize(elmat.Height());
  elmat.Mult(elfun, elvect);
}

void MixedBilinearToNonlinearFormIntegrator::AssembleElementGrad(const mfem::FiniteElement&   el,
                                                                 mfem::ElementTransformation& Tr, const mfem::Vector&,
                                                                 mfem::DenseMatrix&           elmat)
{
  const mfem::FiniteElement& trial_el = *(trial_fes_->FEColl()->FiniteElementForGeometry(Tr.GetGeometryType()));

  A_->AssembleElementMatrix2(trial_el, el, Tr, elmat);
}

SubstitutionNonlinearFormIntegrator::SubstitutionNonlinearFormIntegrator(
    std::shared_ptr<mfem::NonlinearFormIntegrator> R,
    std::function<std::shared_ptr<mfem::Vector>(const mfem::FiniteElement& el, mfem::ElementTransformation& Tr,
                                                const mfem::Vector&)>
        substitute,
    std::function<std::shared_ptr<mfem::DenseMatrix>(const mfem::FiniteElement& el, mfem::ElementTransformation& Tr,
                                                     const mfem::DenseMatrix&)>
        substitute_grad)
    : R_(R), substitute_function_(substitute), substitute_function_grad_(substitute_grad)
{
}

void SubstitutionNonlinearFormIntegrator::AssembleElementVector(const mfem::FiniteElement&   el,
                                                                mfem::ElementTransformation& Tr,
                                                                const mfem::Vector& elfun, mfem::Vector& elvect)
{
  auto substituted = substitute_function_(el, Tr, elfun);
  R_->AssembleElementVector(el, Tr, *substituted, elvect);
}

void SubstitutionNonlinearFormIntegrator::AssembleElementGrad(const mfem::FiniteElement&   el,
                                                              mfem::ElementTransformation& Tr,
                                                              const mfem::Vector& elfun, mfem::DenseMatrix& elmat)
{
  auto substituted = substitute_function_(el, Tr, elfun);
  R_->AssembleElementGrad(el, Tr, *substituted, elmat);
  auto dense_grad_substitute = substitute_function_grad_(el, Tr, elmat);
  elmat                      = *dense_grad_substitute;
}

}  // namespace serac

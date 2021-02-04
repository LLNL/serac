// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/integrators/hyperelastic_traction_integrator.hpp"

namespace serac::mfem_ext {

void HyperelasticTractionIntegrator::AssembleFaceVector(const mfem::FiniteElement& el1, const mfem::FiniteElement&,
                                                        mfem::FaceElementTransformations& Tr, const mfem::Vector& elfun,
                                                        mfem::Vector& elvec)
{
  int dim = el1.GetDim();
  int dof = el1.GetDof();

  shape_.SetSize(dof);
  elvec.SetSize(dim * dof);

  DSh_u_.SetSize(dof, dim);
  DS_u_.SetSize(dof, dim);
  J0i_.SetSize(dim);
  F_.SetSize(dim);
  Finv_.SetSize(dim);

  PMatI_u_.UseExternalData(elfun.GetData(), dof, dim);

  int intorder = 2 * el1.GetOrder() + 3;

  const mfem::IntegrationRule& ir = mfem::IntRules.Get(Tr.FaceGeom, intorder);

  elvec = 0.0;

  mfem::Vector trac(dim);
  mfem::Vector ftrac(dim);
  mfem::Vector nor(dim);
  mfem::Vector fnor(dim);
  mfem::Vector u(dim);
  mfem::Vector fu(dim);

  for (int i = 0; i < ir.GetNPoints(); i++) {
    const mfem::IntegrationPoint& ip = ir.IntPoint(i);
    mfem::IntegrationPoint        eip;
    Tr.Loc1.Transform(ip, eip);

    Tr.Face->SetIntPoint(&ip);

    // Compute traction
    function_.Eval(trac, *Tr.Face, ip);

    if (!compute_on_reference_) {
      Tr.Elem1->SetIntPoint(&eip);
      CalcInverse(Tr.Elem1->Jacobian(), J0i_);

      el1.CalcDShape(eip, DSh_u_);
      Mult(DSh_u_, J0i_, DS_u_);
      MultAtB(PMatI_u_, DS_u_, F_);

      for (int d = 0; d < dim; d++) {
        F_(d, d) += 1.0;
      }

      CalcInverse(F_, Finv_);

      CalcOrtho(Tr.Face->Jacobian(), nor);

      // Normalize vector
      double norm = nor.Norml2();
      nor /= norm;

      Finv_.MultTranspose(nor, fnor);
    }

    el1.CalcShape(eip, shape_);
    for (int j = 0; j < dof; j++) {
      for (int k = 0; k < dim; k++) {
        double residual_contribution = trac(k) * shape_(j) * ip.weight * Tr.Face->Weight();
        if (!compute_on_reference_) {
          residual_contribution *= -1.0 * F_.Det() * fnor.Norml2();
        }
        elvec(dof * k + j) += residual_contribution;
      }
    }
  }
}

void HyperelasticTractionIntegrator::AssembleFaceGrad(const mfem::FiniteElement& el1, const mfem::FiniteElement& el2,
                                                      mfem::FaceElementTransformations& Tr, const mfem::Vector& elfun,
                                                      mfem::DenseMatrix& elmat)
{
  double       diff_step = 1.0e-8;
  mfem::Vector temp_out_1;
  mfem::Vector temp_out_2;
  mfem::Vector temp(elfun.GetData(), elfun.Size());

  elmat.SetSize(elfun.Size(), elfun.Size());

  elmat = 0.0;

  if (!compute_on_reference_) {
    for (int j = 0; j < temp.Size(); j++) {
      temp[j] += diff_step;
      AssembleFaceVector(el1, el2, Tr, temp, temp_out_1);
      temp[j] -= 2.0 * diff_step;
      AssembleFaceVector(el1, el2, Tr, temp, temp_out_2);

      for (int k = 0; k < temp.Size(); k++) {
        elmat(k, j) = (temp_out_1[k] - temp_out_2[k]) / (2.0 * diff_step);
      }
      temp[j] = elfun[j];
    }
  }
}

void HyperelasticPressureIntegrator::AssembleFaceVector(const mfem::FiniteElement& el1, const mfem::FiniteElement&,
                                                        mfem::FaceElementTransformations& Tr, const mfem::Vector&,
                                                        mfem::Vector&                     elvec)
{
  int dim = el1.GetDim();
  int dof = el1.GetDof();

  nor_.SetSize(dim);
  shape_.SetSize(dof);
  elvec.SetSize(dim * dof);

  int intorder = 2 * el1.GetOrder() + 3;

  const mfem::IntegrationRule& ir = mfem::IntRules.Get(Tr.FaceGeom, intorder);

  elvec = 0.0;

  mfem::Vector trac(dim);

  for (int i = 0; i < ir.GetNPoints(); i++) {
    const mfem::IntegrationPoint& ip = ir.IntPoint(i);
    mfem::IntegrationPoint        eip;
    Tr.Loc1.Transform(ip, eip);

    Tr.Face->SetIntPoint(&ip);
    CalcOrtho(Tr.Face->Jacobian(), nor_);

    // Normalize vector
    double norm = nor_.Norml2();
    nor_ /= norm;

    if (compute_on_reference_) {
      fnor_ = nor_;
    } else {
      Tr.Elem1->SetIntPoint(&eip);
      CalcInverse(Tr.Elem1->Jacobian(), J0i_);

      el1.CalcDShape(eip, DSh_u_);
      Mult(DSh_u_, J0i_, DS_u_);
      MultAtB(PMatI_u_, DS_u_, J_);

      CalcInverse(J_, Jinv_);

      Jinv_.MultTranspose(nor_, fnor_);
    }

    el1.CalcShape(eip, shape_);
    fnor_ *= pressure_.Eval(*Tr.Face, ip) * J_.Det();

    for (int j = 0; j < dof; j++) {
      for (int k = 0; k < dim; k++) {
        elvec(dof * k + j) += ip.weight * Tr.Face->Weight() * fnor_(k) * shape_(j);
      }
    }
  }
}

void HyperelasticPressureIntegrator::AssembleFaceGrad(const mfem::FiniteElement& el1, const mfem::FiniteElement& el2,
                                                      mfem::FaceElementTransformations& Tr, const mfem::Vector& elfun,
                                                      mfem::DenseMatrix& elmat)
{
  double       diff_step = 1.0e-8;
  mfem::Vector temp_out_1;
  mfem::Vector temp_out_2;
  mfem::Vector temp(elfun.GetData(), elfun.Size());

  elmat.SetSize(elfun.Size(), elfun.Size());

  elmat = 0.0;

  if (!compute_on_reference_) {
    for (int j = 0; j < temp.Size(); j++) {
      temp[j] += diff_step;
      AssembleFaceVector(el1, el2, Tr, temp, temp_out_1);
      temp[j] -= 2.0 * diff_step;
      AssembleFaceVector(el1, el2, Tr, temp, temp_out_2);

      for (int k = 0; k < temp.Size(); k++) {
        elmat(k, j) = (temp_out_1[k] - temp_out_2[k]) / (2.0 * diff_step);
      }
      temp[j] = elfun[j];
    }
  }
}

}  // namespace serac::mfem_ext

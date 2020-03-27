// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "hyperelastic_traction_integrator.hpp"

void HyperelasticTractionIntegrator::AssembleFaceVector(
    const mfem::FiniteElement &                        el1,
    __attribute__((unused)) const mfem::FiniteElement &el2,
    mfem::FaceElementTransformations &Tr, const mfem::Vector &elfun,
    mfem::Vector &elvec) {
  int dim = el1.GetDim();
  int dof = el1.GetDof();

  shape.SetSize(dof);
  elvec.SetSize(dim * dof);

  DSh_u.SetSize(dof, dim);
  DS_u.SetSize(dof, dim);
  J0i.SetSize(dim);
  F.SetSize(dim);
  Finv.SetSize(dim);

  PMatI_u.UseExternalData(elfun.GetData(), dof, dim);

  int                          intorder = 2 * el1.GetOrder() + 3;
  const mfem::IntegrationRule &ir = mfem::IntRules.Get(Tr.FaceGeom, intorder);

  elvec = 0.0;

  mfem::Vector trac(dim);
  mfem::Vector ftrac(dim);
  mfem::Vector nor(dim);
  mfem::Vector fnor(dim);
  mfem::Vector u(dim);
  mfem::Vector fu(dim);

  for (int i = 0; i < ir.GetNPoints(); i++) {
    const mfem::IntegrationPoint &ip = ir.IntPoint(i);
    mfem::IntegrationPoint        eip;
    Tr.Loc1.Transform(ip, eip);

    Tr.Face->SetIntPoint(&ip);

    CalcOrtho(Tr.Face->Jacobian(), nor);

    // Normalize vector
    double norm = nor.Norml2();
    nor /= norm;

    // Compute traction
    function.Eval(trac, *Tr.Face, ip);

    Tr.Elem1->SetIntPoint(&eip);
    CalcInverse(Tr.Elem1->Jacobian(), J0i);

    el1.CalcDShape(eip, DSh_u);
    Mult(DSh_u, J0i, DS_u);
    MultAtB(PMatI_u, DS_u, F);

    for (int d = 0; d < dim; d++) {
      F(d, d) += 1.0;
    }

    CalcInverse(F, Finv);

    Finv.MultTranspose(nor, fnor);

    el1.CalcShape(eip, shape);
    for (int j = 0; j < dof; j++) {
      for (int k = 0; k < dim; k++) {
        elvec(dof * k + j) -= trac(k) * shape(j) * ip.weight *
                              Tr.Face->Weight() * F.Det() * fnor.Norml2();
      }
    }
  }
}

void HyperelasticTractionIntegrator::AssembleFaceGrad(
    const mfem::FiniteElement &                        el1,
    __attribute__((unused)) const mfem::FiniteElement &el2,
    mfem::FaceElementTransformations &Tr, const mfem::Vector &elfun,
    mfem::DenseMatrix &elmat) {
  double       diff_step = 1.0e-8;
  mfem::Vector temp_out_1;
  mfem::Vector temp_out_2;
  mfem::Vector temp(elfun.GetData(), elfun.Size());

  elmat.SetSize(elfun.Size(), elfun.Size());

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

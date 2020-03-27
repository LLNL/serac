// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "inc_hyperelastic_integrator.hpp"

double IncrementalHyperelasticIntegrator::GetElementEnergy(
    const mfem::FiniteElement &el, mfem::ElementTransformation &Ttr,
    const mfem::Vector &elfun) {
  int    dof = el.GetDof(), dim = el.GetDim();
  double energy;

  DSh.SetSize(dof, dim);
  Jrt.SetSize(dim);
  Jpr.SetSize(dim);
  Jpt.SetSize(dim);
  PMatI.UseExternalData(elfun.GetData(), dof, dim);

  const mfem::IntegrationRule *ir = IntRule;
  if (!ir) {
    ir =
        &(mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 3));  // <---
  }

  energy = 0.0;
  model->SetTransformation(Ttr);
  for (int i = 0; i < ir->GetNPoints(); i++) {
    const mfem::IntegrationPoint &ip = ir->IntPoint(i);
    Ttr.SetIntPoint(&ip);
    CalcInverse(Ttr.Jacobian(), Jrt);

    el.CalcDShape(ip, DSh);
    MultAtB(PMatI, DSh, Jpr);
    Mult(Jpr, Jrt, Jpt);

    for (int d = 0; d < dim; d++) {
      Jpt(d, d) += 1.0;
    }

    energy += ip.weight * Ttr.Weight() * model->EvalW(Jpt);
  }

  return energy;
}

void IncrementalHyperelasticIntegrator::AssembleElementVector(
    const mfem::FiniteElement &el, mfem::ElementTransformation &Ttr,
    const mfem::Vector &elfun, mfem::Vector &elvect) {
  int dof = el.GetDof(), dim = el.GetDim();

  DSh.SetSize(dof, dim);
  DS.SetSize(dof, dim);
  Jrt.SetSize(dim);
  Jpt.SetSize(dim);
  P.SetSize(dim);
  PMatI.UseExternalData(elfun.GetData(), dof, dim);
  elvect.SetSize(dof * dim);
  PMatO.UseExternalData(elvect.GetData(), dof, dim);

  const mfem::IntegrationRule *ir = IntRule;
  if (!ir) {
    ir =
        &(mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 3));  // <---
  }

  elvect = 0.0;
  model->SetTransformation(Ttr);
  for (int i = 0; i < ir->GetNPoints(); i++) {
    const mfem::IntegrationPoint &ip = ir->IntPoint(i);
    Ttr.SetIntPoint(&ip);
    CalcInverse(Ttr.Jacobian(), Jrt);

    el.CalcDShape(ip, DSh);
    Mult(DSh, Jrt, DS);
    MultAtB(PMatI, DS, Jpt);

    for (int d = 0; d < dim; d++) {
      Jpt(d, d) += 1.0;
    }

    model->EvalP(Jpt, P);

    P *= ip.weight * Ttr.Weight();
    AddMultABt(DS, P, PMatO);
  }
}

void IncrementalHyperelasticIntegrator::AssembleElementGrad(
    const mfem::FiniteElement &el, mfem::ElementTransformation &Ttr,
    const mfem::Vector &elfun, mfem::DenseMatrix &elmat) {
  int dof = el.GetDof(), dim = el.GetDim();

  DSh.SetSize(dof, dim);
  DS.SetSize(dof, dim);
  Jrt.SetSize(dim);
  Jpt.SetSize(dim);
  PMatI.UseExternalData(elfun.GetData(), dof, dim);
  elmat.SetSize(dof * dim);

  const mfem::IntegrationRule *ir = IntRule;
  if (!ir) {
    ir =
        &(mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 3));  // <---
  }

  elmat = 0.0;
  model->SetTransformation(Ttr);
  for (int i = 0; i < ir->GetNPoints(); i++) {
    const mfem::IntegrationPoint &ip = ir->IntPoint(i);
    Ttr.SetIntPoint(&ip);
    CalcInverse(Ttr.Jacobian(), Jrt);

    el.CalcDShape(ip, DSh);
    Mult(DSh, Jrt, DS);
    MultAtB(PMatI, DS, Jpt);

    for (int d = 0; d < dim; d++) {
      Jpt(d, d) += 1.0;
    }

    model->AssembleH(Jpt, DS, ip.weight * Ttr.Weight(), elmat);
  }
}

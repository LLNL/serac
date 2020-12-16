// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

<<<<<<< HEAD:src/physics/integrators/displacement_hyperelastic_integrator.cpp
#include "physics/integrators/displacement_hyperelastic_integrator.hpp"
=======
#include "serac/integrators/inc_hyperelastic_integrator.hpp"
>>>>>>> develop:src/serac/integrators/inc_hyperelastic_integrator.cpp

#include "serac/infrastructure/profiling.hpp"

namespace serac {

double DisplacementHyperelasticIntegrator::GetElementEnergy(const mfem::FiniteElement&   el,
                                                           mfem::ElementTransformation& Ttr, const mfem::Vector& elfun)
{
  int dof = el.GetDof(), dim = el.GetDim();

  DSh_.SetSize(dof, dim);
  Jrt_.SetSize(dim);
  Jpr_.SetSize(dim);
  Jpt_.SetSize(dim);
  PMatI_.UseExternalData(elfun.GetData(), dof, dim);

  const mfem::IntegrationRule* ir = IntRule;
  if (!ir) {
    ir = &(mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 3));  // <---
  }

  double energy = 0.0;
  material_.SetTransformation(Ttr);
  for (int i = 0; i < ir->GetNPoints(); i++) {
    const mfem::IntegrationPoint& ip = ir->IntPoint(i);
    Ttr.SetIntPoint(&ip);
    CalcInverse(Ttr.Jacobian(), Jrt_);

    el.CalcDShape(ip, DSh_);
    MultAtB(PMatI_, DSh_, Jpr_);
    Mult(Jpr_, Jrt_, Jpt_);

    for (int d = 0; d < dim; d++) {
      Jpt_(d, d) += 1.0;
    }

    energy += ip.weight * Ttr.Weight() * material_.EvalW(Jpt_);
  }

  return energy;
}

void DisplacementHyperelasticIntegrator::AssembleElementVector(const mfem::FiniteElement&   el,
                                                              mfem::ElementTransformation& Ttr,
                                                              const mfem::Vector& elfun, mfem::Vector& elvect)
{
  int dof = el.GetDof(), dim = el.GetDim();

  DSh_.SetSize(dof, dim);
  DS_.SetSize(dof, dim);
  Jrt_.SetSize(dim);
  Jpt_.SetSize(dim);
  P_.SetSize(dim);
  PMatI_.UseExternalData(elfun.GetData(), dof, dim);
  elvect.SetSize(dof * dim);
  PMatO_.UseExternalData(elvect.GetData(), dof, dim);

  const mfem::IntegrationRule* ir = IntRule;
  if (!ir) {
    ir = &(mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 3));  // <---
  }

  elvect = 0.0;
  material_.SetTransformation(Ttr);
  for (int i = 0; i < ir->GetNPoints(); i++) {
    const mfem::IntegrationPoint& ip = ir->IntPoint(i);
    Ttr.SetIntPoint(&ip);
    CalcInverse(Ttr.Jacobian(), Jrt_);

    el.CalcDShape(ip, DSh_);
    Mult(DSh_, Jrt_, DS_);
    MultAtB(PMatI_, DS_, Jpt_);

    for (int d = 0; d < dim; d++) {
      Jpt_(d, d) += 1.0;
    }

    material_.EvalP(Jpt_, P_);

    P_ *= ip.weight * Ttr.Weight();
    AddMultABt(DS_, P_, PMatO_);
  }
}

void DisplacementHyperelasticIntegrator::AssembleElementGrad(const mfem::FiniteElement&   el,
                                                            mfem::ElementTransformation& Ttr, const mfem::Vector& elfun,
                                                            mfem::DenseMatrix& elmat)
{
  SERAC_MARK_FUNCTION;

  int dof = el.GetDof(), dim = el.GetDim();

  DSh_.SetSize(dof, dim);
  DS_.SetSize(dof, dim);
  Jrt_.SetSize(dim);
  Jpt_.SetSize(dim);
  PMatI_.UseExternalData(elfun.GetData(), dof, dim);
  elmat.SetSize(dof * dim);

  const mfem::IntegrationRule* ir = IntRule;
  if (!ir) {
    ir = &(mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 3));  // <---
  }

  elmat = 0.0;
  material_.SetTransformation(Ttr);
  SERAC_MARK_LOOP_START(ip_loop_id, "IntegrationPt Loop");
  for (int i = 0; i < ir->GetNPoints(); i++) {
    SERAC_MARK_LOOP_ITER(ip_loop_id, i);
    const mfem::IntegrationPoint& ip = ir->IntPoint(i);
    Ttr.SetIntPoint(&ip);
    CalcInverse(Ttr.Jacobian(), Jrt_);

    el.CalcDShape(ip, DSh_);
    Mult(DSh_, Jrt_, DS_);
    MultAtB(PMatI_, DS_, Jpt_);

    for (int d = 0; d < dim; d++) {
      Jpt_(d, d) += 1.0;
    }

    material_.AssembleTangentModuli(Jpt_, DS_, ip.weight * Ttr.Weight(), elmat);
  }
  SERAC_MARK_LOOP_END(ip_loop_id);
}

}  // namespace serac

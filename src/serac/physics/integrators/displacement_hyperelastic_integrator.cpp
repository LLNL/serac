// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/integrators/displacement_hyperelastic_integrator.hpp"

#include "serac/infrastructure/profiling.hpp"
#include "serac/numerics/expr_template_ops.hpp"
#include "serac/numerics/array_4D.hpp"

namespace serac {

void DisplacementHyperelasticIntegrator::CalcDeformationGradient(const mfem::FiniteElement&    el,
                                                                 const mfem::IntegrationPoint& ip,
                                                                 mfem::ElementTransformation&  Ttr)
{
  
  CalcInverse(Ttr.Jacobian(), Jrt_);
  el.CalcDShape(ip, DSh_);
  Mult(DSh_, Jrt_, DS_);

  int dim = Jrt_.Width();

  MultAtB(PMatI_, DS_, H_);

  F_ = H_;
  for (int i=0; i < dim; ++i) {
    F_(i,i) += 1.0;
  }

  mfem::CalcInverse(F_, Finv_);

  // If we're including geometric nonlinearities, integrate on the current configuration
  if (geom_nonlin_) {
    mfem::Mult(DS_, Finv_, B_);
    B_ *= F_.Det();
  } else {
    B_ = DS_;
  }
}

double DisplacementHyperelasticIntegrator::GetElementEnergy(const mfem::FiniteElement&   el,
                                                            mfem::ElementTransformation& Ttr, const mfem::Vector& elfun)
{
  int dof = el.GetDof(), dim = el.GetDim();

  DSh_.SetSize(dof, dim);
  Jrt_.SetSize(dim);
  F_.SetSize(dim);
  H_.SetSize(dim);
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
    CalcDeformationGradient(el, ip, Ttr);
    energy += ip.weight * Ttr.Weight() * material_.EvalW(F_);
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
  B_.SetSize(dof, dim);
  Jrt_.SetSize(dim);
  F_.SetSize(dim);
  Finv_.SetSize(dim);
  H_.SetSize(dim);
  sigma_.SetSize(dim);
  PMatI_.UseExternalData(elfun.GetData(), dof, dim);
  elvect.SetSize(dof * dim);
  PMatO_.UseExternalData(elvect.GetData(), dof, dim);

  const mfem::IntegrationRule* ir = IntRule;
  if (!ir) {
    ir = &(mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 3));  // <---
  }

  elvect = 0.0;
  material_.SetTransformation(Ttr);

  PMatO_ = 0.0;

  for (int i = 0; i < ir->GetNPoints(); i++) {
    const mfem::IntegrationPoint& ip = ir->IntPoint(i);
    Ttr.SetIntPoint(&ip);
    CalcDeformationGradient(el, ip, Ttr);

    material_.EvalStress(F_, sigma_);

    sigma_ *= ip.weight * Ttr.Weight();
    mfem::AddMult(B_, sigma_, PMatO_);
  }
}

void DisplacementHyperelasticIntegrator::AssembleElementGrad(const mfem::FiniteElement&   el,
                                                             mfem::ElementTransformation& Ttr,
                                                             const mfem::Vector& elfun, mfem::DenseMatrix& elmat)
{
  /*
  double       diff_step = 1.0e-8;
  mfem::Vector temp_out_1;
  mfem::Vector temp_out_2;
  mfem::Vector temp(elfun.GetData(), elfun.Size());

  elmat.SetSize(elfun.Size(), elfun.Size());

  for (int j = 0; j < temp.Size(); j++) {
    temp[j] += diff_step;
    AssembleElementVector(el, Ttr, temp, temp_out_1);
    temp[j] -= 2.0 * diff_step;
    AssembleElementVector(el, Ttr, temp, temp_out_2);

    for (int k = 0; k < temp.Size(); k++) {
      elmat(k, j) = (temp_out_1[k] - temp_out_2[k]) / (2.0 * diff_step);
    }
    temp[j] = elfun[j];
  }
  */
  SERAC_MARK_FUNCTION;

  int dof = el.GetDof(), dim = el.GetDim();

  DSh_.SetSize(dof, dim);
  DS_.SetSize(dof, dim);
  B_.SetSize(dof, dim);
  Jrt_.SetSize(dim);
  F_.SetSize(dim);
  Finv_.SetSize(dim);
  H_.SetSize(dim);
  sigma_.SetSize(dim);
  PMatI_.UseExternalData(elfun.GetData(), dof, dim);
  elmat.SetSize(dof * dim);
  C_.SetSize(dim, dim, dim, dim);

  const mfem::IntegrationRule* ir = IntRule;
  if (!ir) {
    ir = &(mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 3));  // <---
  }

  elmat = 0.0;
  material_.SetTransformation(Ttr);
  SERAC_MARK_LOOP_START(ip_loop_id, "IntegrationPt Loop");

  for (int ip_num = 0; ip_num < ir->GetNPoints(); ip_num++) {
    SERAC_MARK_LOOP_ITER(ip_loop_id, i);
    const mfem::IntegrationPoint& ip = ir->IntPoint(ip_num);
    Ttr.SetIntPoint(&ip);
    CalcDeformationGradient(el, ip, Ttr);

    material_.AssembleTangentModuli(F_, C_);
    double J = F_.Det();

    // material stiffness
    for (int a = 0; a < dof; ++a) {
      for (int i = 0; i < dim; ++i) {
        for (int b = 0; b < dof; ++b) {
          for (int k = 0; k < dim; ++k) {
            for (int j = 0; j < dim; ++j) {
              for (int l = 0; l < dim; ++l) {
                elmat(i * dof + a, k * dof + b) += C_(i,j,k,l) * B_(a,j) * B_(b,l) * J * ip.weight * Ttr.Weight();
              }
            }
          }
        }
      }
    }
    // geometric stiffness
    if (geom_nonlin_) {
      material_.EvalStress(F_, sigma_);
      for (int a = 0; a < dof; ++a) {
        for (int i = 0; i < dim; ++i) {
          for (int b = 0; b < dof; ++b) {
            for (int k = 0; k < dim; ++k) {
              for (int j = 0; j < dim; ++j) {
                elmat(i * dof + a, k * dof + b) -=  J * sigma_(i,j) * B_(a,k) * B_(b,j) * ip.weight * Ttr.Weight();
              }
            }
          }
        }
      }
    }
  }
  SERAC_MARK_LOOP_END(ip_loop_id);
}

}  // namespace serac

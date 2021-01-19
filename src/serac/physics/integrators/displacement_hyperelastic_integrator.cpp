// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/integrators/displacement_hyperelastic_integrator.hpp"

#include "serac/infrastructure/profiling.hpp"
#include "serac/numerics/expr_template_ops.hpp"
#include "serac/numerics/voigt_tensor.hpp"

namespace serac {

void DisplacementHyperelasticIntegrator::CalcDeformationGradient(const mfem::FiniteElement&    el,
                                                                 const mfem::IntegrationPoint& ip,
                                                                 mfem::ElementTransformation&  Ttr)
{
  CalcInverse(Ttr.Jacobian(), Jrt_);
  el.CalcDShape(ip, DSh_);
  Mult(DSh_, Jrt_, DS_);

  MultAtB(PMatI_, DS_, H_);

  mfem::Add(1.0, H_, 1.0, eye_, F_);
  HT_.Transpose(H_);

  C_ = eye_;

  C_ += H_;
  C_ += HT_;

  mfem::AddMult(HT_, H_, C_);
}

void DisplacementHyperelasticIntegrator::CalcBMatrix()
{
  int dof = DSh_.Height();
  int dim = DSh_.Width();

  if (geom_nonlin_) {
    for (int n = 0; n < dof; ++n) {
      for (int j = 0; j < dim; ++j) {
        for (int i = 0; i < dim; ++i) {
          B_0_[n](i, j) = DS_(n, i) * F_(j, i);
        }
        for (int i = 0; i < (int)shear_terms_.size(); ++i) {
          B_0_[n](dim + i, j) = DS_(n, shear_terms_[i].first) * F_(j, shear_terms_[i].second) +
                                DS_(n, shear_terms_[i].second) * F_(j, shear_terms_[i].first);
        }
      }
    }
  } else {
    for (int n = 0; n < dof; ++n) {
      for (int i = 0; i < dim; ++i) {
        B_0_[n](i, i) = DS_(n, i);
      }
    }
  }
}

double DisplacementHyperelasticIntegrator::GetElementEnergy(const mfem::FiniteElement&   el,
                                                            mfem::ElementTransformation& Ttr, const mfem::Vector& elfun)
{
  int dof = el.GetDof(), dim = el.GetDim();

  DSh_.SetSize(dof, dim);
  Jrt_.SetSize(dim);
  F_.SetSize(dim);
  C_.SetSize(dim);
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
    energy += ip.weight * Ttr.Weight() * material_.EvalW(C_);
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
  F_.SetSize(dim);
  C_.SetSize(dim);
  S_.SetSize(dim);
  PMatI_.UseExternalData(elfun.GetData(), dof, dim);
  elvect.SetSize(dof * dim);
  PMatO_.UseExternalData(elvect.GetData(), dof, dim);
  force_.SetSize(dim);

  B_0_.resize(dof);
  for (int i = 0; i < dof; ++i) {
    B_0_[i].SetSize(dim + shear_terms_.size(), dim);
  }

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
    CalcBMatrix();

    material_.EvalPK2(C_, S_);

    for (int n = 0; n < dof; ++n) {
      B_0_[n].MultTranspose(S_, force_);
      force_ *= ip.weight * Ttr.Weight();
      PMatO_.SetRow(n, force_);
    }
  }
}

void DisplacementHyperelasticIntegrator::AssembleElementGrad(const mfem::FiniteElement&   el,
                                                             mfem::ElementTransformation& Ttr,
                                                             const mfem::Vector& elfun, mfem::DenseMatrix& elmat)
{
  SERAC_MARK_FUNCTION;

  int dof = el.GetDof(), dim = el.GetDim();

  DSh_.SetSize(dof, dim);
  DS_.SetSize(dof, dim);
  Jrt_.SetSize(dim);
  F_.SetSize(dim);
  PMatI_.UseExternalData(elfun.GetData(), dof, dim);
  elmat.SetSize(dof * dim);
  C_.SetSize(dim);
  temp_.SetSize(dim + shear_terms_.size(), dim);
  K_.SetSize(dim);

  B_0_.resize(dof);
  for (int i = 0; i < dof; ++i) {
    B_0_[i].SetSize(dim + shear_terms_.size(), dim);
  }

  B_0T.SetSize(dim, dim + shear_terms_.size());

  T_.SetSize(dim + shear_terms_.size());

  const mfem::IntegrationRule* ir = IntRule;
  if (!ir) {
    ir = &(mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 3));  // <---
  }

  mfem::DenseMatrix geom_stiff(dof, dof);

  elmat = 0.0;
  material_.SetTransformation(Ttr);
  SERAC_MARK_LOOP_START(ip_loop_id, "IntegrationPt Loop");
  for (int i = 0; i < ir->GetNPoints(); i++) {
    SERAC_MARK_LOOP_ITER(ip_loop_id, i);
    const mfem::IntegrationPoint& ip = ir->IntPoint(i);
    Ttr.SetIntPoint(&ip);
    CalcDeformationGradient(el, ip, Ttr);
    CalcBMatrix();

    material_.AssembleTangentModuli(C_, T_);

    // material stiffness
    for (int n = 0; n < dof; ++n) {
      B_0T.Transpose(B_0_[n]);
      for (int m = 0; m < dof; ++m) {
        mfem::Mult(temp_, B_0_[m], K_);
        for (int i = 0; i < dim; ++i) {
          for (int j = 0; j < dim; ++j) {
            elmat(i * dof + n, j * dof + m) += K_(i, j) * ip.weight * Ttr.Weight();
          }
        }
      }
    }
    // geometric stiffness
    if (geom_nonlin_) {
      material_.EvalPK2(C_, S_);
      getTensorFromVoigtVector(shear_terms_, S_, S_mat_);
      for (int n = 0; n < dof; ++n) {
        for (int m = 0; m < dof; ++m) {
          for (int r = 0; r < dim; ++r) {
            for (int i = 0; i < dim; ++i) {
              for (int j = 0; j < dim; ++j) {
                elmat(r * dof + n, r * dof + m) += DS_(n, j) * S_mat_(j, i) * DS_(m, i) * ip.weight * Ttr.Weight();
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

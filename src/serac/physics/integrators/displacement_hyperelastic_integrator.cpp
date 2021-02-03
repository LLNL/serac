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
  // Calculate the reference to stress-free transformation
  CalcInverse(Ttr.Jacobian(), Jrt_);

  // Calculate the derivatives of the shape functions in the reference space
  el.CalcDShape(ip, DSh_);

  // Calculate the derivatives of the shape functions in the stress free configuration
  Mult(DSh_, Jrt_, DS_);

  int dim = Jrt_.Width();

  // Calculate the displacement gradient using the current DOFs
  MultAtB(PMatI_, DS_, H_);

  // Add the identity matrix to calculate the deformation gradient F_
  F_ = H_;
  for (int i = 0; i < dim; ++i) {
    F_(i, i) += 1.0;
  }

  // Calculate the inverse of the deformation gradient
  mfem::CalcInverse(F_, Finv_);

  // Calculate the B matrix (dN/Dx where x is the current configuration)
  // If we're including geometric nonlinearities, integrate on the current configuration
  if (geom_nonlin_) {
    mfem::Mult(DS_, Finv_, B_);
    // Calculate the determinant of the deformation gradient
    J_ = F_.Det();
  } else {
    B_ = DS_;
    J_ = 1.0;
  }
}

double DisplacementHyperelasticIntegrator::GetElementEnergy(const mfem::FiniteElement&   el,
                                                            mfem::ElementTransformation& Ttr, const mfem::Vector& elfun)
{
  int dof = el.GetDof(), dim = el.GetDim();

  // Reshape the state vector
  PMatI_.UseExternalData(elfun.GetData(), dof, dim);

  const mfem::IntegrationRule* ir = IntRule;
  if (!ir) {
    ir = &(mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 3));  // <---
  }

  double energy = 0.0;
  material_.SetTransformation(Ttr);
  for (int i = 0; i < ir->GetNPoints(); i++) {
    // Set the current integration point
    const mfem::IntegrationPoint& ip = ir->IntPoint(i);
    Ttr.SetIntPoint(&ip);

    // Calculate the deformation gradent and accumulate the strain energy at the current integration point
    CalcDeformationGradient(el, ip, Ttr);
    energy += J_ * ip.weight * Ttr.Weight() * material_.EvalStrainEnergy(F_);
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

  // Reshape the input state and residual vectors
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
    // Set the current integration point
    const mfem::IntegrationPoint& ip = ir->IntPoint(i);
    Ttr.SetIntPoint(&ip);

    // Calculate the deformation gradient at the current integration point
    CalcDeformationGradient(el, ip, Ttr);

    // Evaluate the Cauchy stress using the calculated deformation gradient
    material_.EvalStress(F_, sigma_);

    // Accumulate the residual using the Cauchy stress and the B matrix
    sigma_ *= J_ * ip.weight * Ttr.Weight();
    mfem::AddMult(B_, sigma_, PMatO_);
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
    // Set the integration point and calculate the deformation gradient
    SERAC_MARK_LOOP_ITER(ip_loop_id, i);
    const mfem::IntegrationPoint& ip = ir->IntPoint(ip_num);
    Ttr.SetIntPoint(&ip);
    CalcDeformationGradient(el, ip, Ttr);

    // Assemble the spatial tangent moduli at the current integration point
    material_.AssembleTangentModuli(F_, C_);

    // Accumulate the material stiffness using the spatial tangent moduli and the B matrix
    for (int a = 0; a < dof; ++a) {
      for (int i = 0; i < dim; ++i) {
        for (int b = 0; b < dof; ++b) {
          for (int k = 0; k < dim; ++k) {
            for (int j = 0; j < dim; ++j) {
              for (int l = 0; l < dim; ++l) {
                elmat(i * dof + a, k * dof + b) += C_(i, j, k, l) * B_(a, j) * B_(b, l) * ip.weight * Ttr.Weight();
              }
            }
          }
        }
      }
    }

    // Accumulate the geometric stiffness if desired
    if (geom_nonlin_) {
      material_.EvalStress(F_, sigma_);
      for (int a = 0; a < dof; ++a) {
        for (int i = 0; i < dim; ++i) {
          for (int b = 0; b < dof; ++b) {
            for (int k = 0; k < dim; ++k) {
              for (int j = 0; j < dim; ++j) {
                elmat(i * dof + a, k * dof + b) -= J_ * sigma_(i, j) * B_(a, k) * B_(b, j) * ip.weight * Ttr.Weight();
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

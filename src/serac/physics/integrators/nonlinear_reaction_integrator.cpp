// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/integrators/nonlinear_reaction_integrator.hpp"

namespace serac::thermal::mfem_ext {

void NonlinearReactionIntegrator::AssembleElementVector(const mfem::FiniteElement&   element,
                                                        mfem::ElementTransformation& parent_to_reference_transformation,
                                                        const mfem::Vector& state_vector, mfem::Vector& residual_vector)
{
  int dof = element.GetDof();

  shape_.SetSize(dof);  // vector of size dof
  residual_vector.SetSize(dof);
  residual_vector = 0.0;

  const mfem::IntegrationRule* ir = IntRule;
  if (ir == nullptr) {
    ir = &mfem::IntRules.Get(element.GetGeomType(), 2 * element.GetOrder() + 3);
  }

  for (int i = 0; i < ir->GetNPoints(); i++) {
    const mfem::IntegrationPoint& ip = ir->IntPoint(i);

    parent_to_reference_transformation.SetIntPoint(&ip);

    // Calculate the derivatives of the shape functions in the reference space
    element.CalcShape(ip, shape_);
    double temp = shape_ * state_vector;

    double source = reaction_(temp);

    residual_vector.Add(ip.weight * parent_to_reference_transformation.Weight() * source, shape_);
  }
}

void NonlinearReactionIntegrator::AssembleElementGrad(const mfem::FiniteElement&   element,
                                                      mfem::ElementTransformation& parent_to_reference_transformation,
                                                      const mfem::Vector&          state_vector,
                                                      mfem::DenseMatrix&           stiffness_matrix)
{
  int dof = element.GetDof();

  shape_.SetSize(dof);  // vector of size dof
  stiffness_matrix.SetSize(dof);

  stiffness_matrix = 0.0;

  const mfem::IntegrationRule* ir = IntRule;
  if (ir == nullptr) {
    ir = &mfem::IntRules.Get(element.GetGeomType(), 2 * element.GetOrder() + 3);
  }

  for (int i = 0; i < ir->GetNPoints(); i++) {
    const mfem::IntegrationPoint& ip = ir->IntPoint(i);

    parent_to_reference_transformation.SetIntPoint(&ip);

    // Calculate the derivatives of the shape functions in the reference space
    element.CalcShape(ip, shape_);
    double temp = shape_ * state_vector;

    double d_source = d_reaction_(temp);

    mfem::AddMult_a_VVt(d_source * ip.weight * parent_to_reference_transformation.Weight(), shape_, stiffness_matrix);
  }
}

}  // namespace serac::thermal::mfem_ext

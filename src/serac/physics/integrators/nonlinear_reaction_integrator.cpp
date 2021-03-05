// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/integrators/nonlinear_reaction_integrator.hpp"

namespace serac::mfem_ext {

void NonlinearReactionIntegrator::AssembleElementVector(const mfem::FiniteElement&   element,
                                                        mfem::ElementTransformation& parent_to_reference_transformation,
                                                        const mfem::Vector& state_vector, mfem::Vector& residual_vector)
{
  int dof = element.GetDof();

  // Ensure the data containers are properly sized
  shape_.SetSize(dof);
  residual_vector.SetSize(dof);

  // Initialize the residual
  residual_vector = 0.0;

  // Determine the integration rule from the element order
  const mfem::IntegrationRule* ir = IntRule;
  if (ir == nullptr) {
    ir = &mfem::IntRules.Get(element.GetGeomType(), 2 * element.GetOrder() + 3);
  }

  for (int i = 0; i < ir->GetNPoints(); i++) {
    // Set the integration point
    const mfem::IntegrationPoint& ip = ir->IntPoint(i);
    parent_to_reference_transformation.SetIntPoint(&ip);

    // Calculate the shape functions in the reference space
    element.CalcShape(ip, shape_);

    // Calculate the temperature at the integration point
    double temp = shape_ * state_vector;

    // Evaluate the scaling coefficent
    double scale = scale_.Eval(parent_to_reference_transformation, ip);

    // Calculate the reaction term from the current temperature
    double source = reaction_(temp);

    // Accumulate the residual contribution
    residual_vector.Add(ip.weight * parent_to_reference_transformation.Weight() * source * scale, shape_);
  }
}

void NonlinearReactionIntegrator::AssembleElementGrad(const mfem::FiniteElement&   element,
                                                      mfem::ElementTransformation& parent_to_reference_transformation,
                                                      const mfem::Vector&          state_vector,
                                                      mfem::DenseMatrix&           stiffness_matrix)
{
  int dof = element.GetDof();

  // Ensure the data containers are properly sized
  shape_.SetSize(dof);
  stiffness_matrix.SetSize(dof);

  // Initialize the stiffness matrix
  stiffness_matrix = 0.0;

  // Determine the integration rule from the order of the element
  const mfem::IntegrationRule* ir = IntRule;
  if (ir == nullptr) {
    ir = &mfem::IntRules.Get(element.GetGeomType(), 2 * element.GetOrder() + 3);
  }

  for (int i = 0; i < ir->GetNPoints(); i++) {
    // Set the current integration point
    const mfem::IntegrationPoint& ip = ir->IntPoint(i);
    parent_to_reference_transformation.SetIntPoint(&ip);

    // Calculate the shape functions in the reference space
    element.CalcShape(ip, shape_);

    // Calculate the temperature at the current integration point
    double temp = shape_ * state_vector;

    // Evaluate the scaling coefficent
    double scale = scale_.Eval(parent_to_reference_transformation, ip);

    // Calculate the derivative of the nonlinear reaction at the current integration point
    double d_source = d_reaction_(temp);

    // Accumulate the stiffness matrix contributions
    mfem::AddMult_a_VVt(d_source * scale * ip.weight * parent_to_reference_transformation.Weight(), shape_,
                        stiffness_matrix);
  }
}

}  // namespace serac::mfem_ext

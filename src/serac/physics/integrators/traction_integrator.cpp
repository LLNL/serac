// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/integrators/traction_integrator.hpp"

namespace serac::mfem_ext {

void TractionIntegrator::AssembleFaceVector(const mfem::FiniteElement&        element_1, const mfem::FiniteElement&,
                                            mfem::FaceElementTransformations& parent_to_reference_face_transformation,
                                            const mfem::Vector&               input_state_vector,
                                            mfem::Vector&                     output_residual_vector)
{
  // Get the dimension and number of degrees of freedom from the current element
  int dim = element_1.GetDim();
  int dof = element_1.GetDof();

  // Ensure the working data structures are sized appropriately
  shape_.SetSize(dof);
  output_residual_vector.SetSize(dim * dof);
  dN_dxi_.SetSize(dof, dim);
  dN_dX_.SetSize(dof, dim);
  dxi_dX_.SetSize(dim);
  F_.SetSize(dim);
  Finv_.SetSize(dim);
  traction_vector_.SetSize(dim);
  nor_.SetSize(dim);
  fnor_.SetSize(dim);

  // Reshape the input state as a matrix (dof x dim)
  input_state_matrix_.UseExternalData(input_state_vector.GetData(), dof, dim);

  // Calculate an appropriate integration rule for the element order
  int                          intorder = 2 * element_1.GetOrder() + 3;
  const mfem::IntegrationRule& ir = mfem::IntRules.Get(parent_to_reference_face_transformation.FaceGeom, intorder);

  // Initialize the residual output
  output_residual_vector = 0.0;

  for (int i = 0; i < ir.GetNPoints(); i++) {
    // Set the current integration point
    const mfem::IntegrationPoint& ip = ir.IntPoint(i);
    mfem::IntegrationPoint        eip;
    parent_to_reference_face_transformation.Loc1.Transform(ip, eip);
    parent_to_reference_face_transformation.Face->SetIntPoint(&ip);

    // Compute the traction at the integration point
    traction_.Eval(traction_vector_, *parent_to_reference_face_transformation.Face, ip);

    // If computing on the deformed configuration, transform the normal vector appropriately
    if (!compute_on_reference_) {
      // Calculate the deformation gradient and its inverse
      parent_to_reference_face_transformation.Elem1->SetIntPoint(&eip);
      CalcInverse(parent_to_reference_face_transformation.Elem1->Jacobian(), dxi_dX_);
      element_1.CalcDShape(eip, dN_dxi_);
      Mult(dN_dxi_, dxi_dX_, dN_dX_);
      MultAtB(input_state_matrix_, dN_dX_, F_);

      for (int d = 0; d < dim; d++) {
        F_(d, d) += 1.0;
      }

      CalcInverse(F_, Finv_);

      // Calculate the normal vector in the reference configuration
      CalcOrtho(parent_to_reference_face_transformation.Face->Jacobian(), nor_);

      // Normalize vector
      double norm = nor_.Norml2();
      nor_ /= norm;

      // Calculate F inverse time the normal vector
      Finv_.MultTranspose(nor_, fnor_);
    }

    // Calculate the shape functions
    element_1.CalcShape(eip, shape_);
    for (int j = 0; j < dof; j++) {
      for (int k = 0; k < dim; k++) {
        double residual_contribution =
            -1.0 * traction_vector_(k) * shape_(j) * ip.weight * parent_to_reference_face_transformation.Face->Weight();
        if (!compute_on_reference_) {
          residual_contribution *= F_.Det() * fnor_.Norml2();
        }
        output_residual_vector(dof * k + j) += residual_contribution;
      }
    }
  }
}

void TractionIntegrator::AssembleFaceGrad(const mfem::FiniteElement& element_1, const mfem::FiniteElement& element_2,
                                          mfem::FaceElementTransformations& parent_to_reference_face_transformation,
                                          const mfem::Vector& input_state_vector, mfem::DenseMatrix& stiffness_matrix)
{
  double       diff_step = 1.0e-8;
  mfem::Vector temp_out_1;
  mfem::Vector temp_out_2;
  mfem::Vector temp(input_state_vector.GetData(), input_state_vector.Size());

  stiffness_matrix.SetSize(input_state_vector.Size(), input_state_vector.Size());
  stiffness_matrix = 0.0;

  // If computing on the deformed configuration calculate the stiffness contributions via finite difference
  // Otherwise, the contributuion is zero
  if (!compute_on_reference_) {
    for (int j = 0; j < temp.Size(); j++) {
      temp[j] += diff_step;
      AssembleFaceVector(element_1, element_2, parent_to_reference_face_transformation, temp, temp_out_1);
      temp[j] -= 2.0 * diff_step;
      AssembleFaceVector(element_1, element_2, parent_to_reference_face_transformation, temp, temp_out_2);

      for (int k = 0; k < temp.Size(); k++) {
        stiffness_matrix(k, j) = (temp_out_1[k] - temp_out_2[k]) / (2.0 * diff_step);
      }
      temp[j] = input_state_vector[j];
    }
  }
}

void PressureIntegrator::AssembleFaceVector(const mfem::FiniteElement&        element_1, const mfem::FiniteElement&,
                                            mfem::FaceElementTransformations& parent_to_reference_face_transformation,
                                            const mfem::Vector&               input_state_vector,
                                            mfem::Vector&                     output_residual_vector)
{
  // Get the dimension and number of degrees of freedom from the current element
  int dim = element_1.GetDim();
  int dof = element_1.GetDof();

  // Ensure the working data structures are sized appropriately
  shape_.SetSize(dof);
  output_residual_vector.SetSize(dim * dof);
  dN_dxi_.SetSize(dof, dim);
  dN_dX_.SetSize(dof, dim);
  dxi_dX_.SetSize(dim);
  F_.SetSize(dim);
  Finv_.SetSize(dim);
  nor_.SetSize(dim);
  fnor_.SetSize(dim);

  // Reshape the input state as a matrix (dof x dim)
  input_state_matrix_.UseExternalData(input_state_vector.GetData(), dof, dim);

  // Calculate an appropriate integration rule for the element order
  int                          intorder = 2 * element_1.GetOrder() + 3;
  const mfem::IntegrationRule& ir = mfem::IntRules.Get(parent_to_reference_face_transformation.FaceGeom, intorder);

  // Initialize the residual output
  output_residual_vector = 0.0;

  for (int i = 0; i < ir.GetNPoints(); i++) {
    // Set the current integration point
    const mfem::IntegrationPoint& ip = ir.IntPoint(i);
    mfem::IntegrationPoint        eip;
    parent_to_reference_face_transformation.Loc1.Transform(ip, eip);
    parent_to_reference_face_transformation.Face->SetIntPoint(&ip);

    // Calculate the normal vector in the reference configuration
    CalcOrtho(parent_to_reference_face_transformation.Face->Jacobian(), nor_);
    double norm = nor_.Norml2();
    nor_ /= norm;

    if (compute_on_reference_) {
      fnor_ = nor_;
    } else {
      // Calculate the deformation gradient and its inverse
      parent_to_reference_face_transformation.Elem1->SetIntPoint(&eip);
      CalcInverse(parent_to_reference_face_transformation.Elem1->Jacobian(), dxi_dX_);
      element_1.CalcDShape(eip, dN_dxi_);
      Mult(dN_dxi_, dxi_dX_, dN_dX_);
      MultAtB(input_state_matrix_, dN_dX_, F_);

      for (int d = 0; d < dim; d++) {
        F_(d, d) += 1.0;
      }

      CalcInverse(F_, Finv_);

      Finv_.MultTranspose(nor_, fnor_);
      fnor_ *= F_.Det();
    }

    element_1.CalcShape(eip, shape_);
    fnor_ *= pressure_.Eval(*parent_to_reference_face_transformation.Face, ip);

    for (int j = 0; j < dof; j++) {
      for (int k = 0; k < dim; k++) {
        output_residual_vector(dof * k + j) +=
            ip.weight * parent_to_reference_face_transformation.Face->Weight() * fnor_(k) * shape_(j);
      }
    }
  }
}

void PressureIntegrator::AssembleFaceGrad(const mfem::FiniteElement& element_1, const mfem::FiniteElement& element_2,
                                          mfem::FaceElementTransformations& parent_to_reference_face_transformation,
                                          const mfem::Vector& input_state_vector, mfem::DenseMatrix& stiffness_matrix)
{
  double       diff_step = 1.0e-8;
  mfem::Vector temp_out_1;
  mfem::Vector temp_out_2;
  mfem::Vector temp(input_state_vector.GetData(), input_state_vector.Size());

  stiffness_matrix.SetSize(input_state_vector.Size(), input_state_vector.Size());

  stiffness_matrix = 0.0;

  // If computing on the deformed configuration calculate the stiffness contributions via finite difference
  // Otherwise, the contribution is zero
  if (!compute_on_reference_) {
    for (int j = 0; j < temp.Size(); j++) {
      temp[j] += diff_step;
      AssembleFaceVector(element_1, element_2, parent_to_reference_face_transformation, temp, temp_out_1);
      temp[j] -= 2.0 * diff_step;
      AssembleFaceVector(element_1, element_2, parent_to_reference_face_transformation, temp, temp_out_2);

      for (int k = 0; k < temp.Size(); k++) {
        stiffness_matrix(k, j) = (temp_out_1[k] - temp_out_2[k]) / (2.0 * diff_step);
      }
      temp[j] = input_state_vector[j];
    }
  }
}

}  // namespace serac::mfem_ext

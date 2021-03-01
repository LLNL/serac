// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/integrators/traction_integrator.hpp"
#include "serac/physics/utilities/physics_utils.hpp"

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
  du_dX_.SetSize(dim);
  dxi_dX_.SetSize(dim);
  F_.SetSize(dim);
  Finv_.SetSize(dim);
  traction_vector_.SetSize(dim);
  reference_normal_.SetSize(dim);
  current_normal_.SetSize(dim);

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

      // Calculate the reference to stress-free transformation
      CalcInverse(parent_to_reference_face_transformation.Elem1->Jacobian(), dxi_dX_);

      // Calculate the derivatives of the shape functions in the reference space
      element_1.CalcDShape(eip, dN_dxi_);

      // Calculate the derivatives of the shape functions in the stress free configuration
      Mult(dN_dxi_, dxi_dX_, dN_dX_);

      // Calculate the displacement gradient using the current DOFs
      MultAtB(input_state_matrix_, dN_dX_, du_dX_);

      solid_util::calcDeformationGradient(du_dX_, F_);

      CalcInverse(F_, Finv_);

      // Calculate the normal vector in the reference configuration
      CalcOrtho(parent_to_reference_face_transformation.Face->Jacobian(), reference_normal_);

      // Normalize vector
      double norm = reference_normal_.Norml2();
      reference_normal_ /= norm;

      // Calculate F inverse time the normal vector
      Finv_.MultTranspose(reference_normal_, current_normal_);
    }

    // Calculate the shape functions
    element_1.CalcShape(eip, shape_);
    for (int j = 0; j < dof; j++) {
      for (int k = 0; k < dim; k++) {
        double residual_contribution =
            -1.0 * traction_vector_(k) * shape_(j) * ip.weight * parent_to_reference_face_transformation.Face->Weight();
        if (!compute_on_reference_) {
          residual_contribution *= F_.Det() * current_normal_.Norml2();
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
  //
  // NOTE: This is not performant and should be replaced by the upcoming weak_form functionality
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
  du_dX_.SetSize(dim);
  F_.SetSize(dim);
  Finv_.SetSize(dim);
  reference_normal_.SetSize(dim);
  current_normal_.SetSize(dim);

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
    CalcOrtho(parent_to_reference_face_transformation.Face->Jacobian(), reference_normal_);
    double norm = reference_normal_.Norml2();
    reference_normal_ /= norm;

    if (compute_on_reference_) {
      current_normal_ = reference_normal_;
    } else {
      // Calculate the deformation gradient and its inverse
      parent_to_reference_face_transformation.Elem1->SetIntPoint(&eip);

      // Calculate the reference to stress-free transformation
      CalcInverse(parent_to_reference_face_transformation.Elem1->Jacobian(), dxi_dX_);

      // Calculate the derivatives of the shape functions in the reference space
      element_1.CalcDShape(eip, dN_dxi_);

      // Calculate the derivatives of the shape functions in the stress free configuration
      Mult(dN_dxi_, dxi_dX_, dN_dX_);

      // Calculate the displacement gradient using the current DOFs
      MultAtB(input_state_matrix_, dN_dX_, du_dX_);

      solid_util::calcDeformationGradient(du_dX_, F_);

      CalcInverse(F_, Finv_);

      // Note that this is not normalized due to the use of Nanson's rule
      //
      // da n = J dA F^-T N
      //
      // where da and n are in the current configuration and dA and N are
      // in the reference configuration
      Finv_.MultTranspose(reference_normal_, current_normal_);
      current_normal_ *= F_.Det();
    }

    element_1.CalcShape(eip, shape_);
    current_normal_ *= pressure_.Eval(*parent_to_reference_face_transformation.Face, ip);

    for (int j = 0; j < dof; j++) {
      for (int k = 0; k < dim; k++) {
        // current_normal has the area transformations per Nanson's formula and includes the applied pressure
        output_residual_vector(dof * k + j) +=
            ip.weight * parent_to_reference_face_transformation.Face->Weight() * current_normal_(k) * shape_(j);
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
  //
  // NOTE: This is not performant and should be replaced by the upcoming weak_form functionality
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

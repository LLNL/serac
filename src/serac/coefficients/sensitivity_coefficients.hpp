// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file sensitivity_coefficients.hpp
 *
 * @brief Coefficients to help with computing sensitivities
 */

#pragma once

#include "serac/physics/utilities/finite_element_state.hpp"
#include "serac/physics/materials/hyperelastic_material.hpp"
#include "serac/coefficients/sensitivity_coefficients.hpp"

#include "mfem.hpp"

namespace serac::mfem_ext {

class ShearSensitivityCoefficient : public mfem::Coefficient {
public:
  ShearSensitivityCoefficient(FiniteElementState& displacement, FiniteElementState& adjoint_displacement_,
                              LinearElasticMaterial& linear_mat);

  ShearSensitivityCoefficient() = delete;

  /// Evaluate the coefficient at @a ip.
  virtual double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip);

private:
  FiniteElementState& displacement_;

  FiniteElementState& adjoint_displacement_;

  LinearElasticMaterial& material_;

  mfem::DenseMatrix du_dX_;

  mfem::DenseMatrix dp_dX_;

  mfem::DenseMatrix d_sigma_d_shear_;

  mfem::DenseMatrix adjoint_strain_;

  mfem::DenseMatrix F_;

  int dim_;
};

class BulkSensitivityCoefficient : public mfem::Coefficient {
public:
  BulkSensitivityCoefficient(FiniteElementState& displacement, FiniteElementState& adjoint_displacement_,
                             LinearElasticMaterial& linear_mat);

  BulkSensitivityCoefficient() = delete;

  /// Evaluate the coefficient at @a ip.
  virtual double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip);

private:
  FiniteElementState& displacement_;

  FiniteElementState& adjoint_displacement_;

  LinearElasticMaterial& material_;

  mfem::DenseMatrix du_dX_;

  mfem::DenseMatrix dp_dX_;

  mfem::DenseMatrix d_sigma_d_bulk_;

  mfem::DenseMatrix adjoint_strain_;

  mfem::DenseMatrix F_;

  int dim_;
};

}  // namespace serac::mfem_ext

// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file hyperelastic_material.hpp
 *
 * @brief The hyperelastic material models for the solid module
 */

#pragma once

#include "mfem.hpp"
#include "serac/physics/utilities/finite_element_state.hpp"

namespace serac {

/**
 * @brief Abstract interface class for a generic hyperelastic material
 *
 */
class ThermalExpansionMaterial {
public:
  /**
   * @brief Construct a new Hyperelastic Material object
   *
   */
  ThermalExpansionMaterial() : parent_to_reference_transformation_(nullptr) {}

  /**
   * @brief Destroy the Hyperelastic Material object
   *
   */
  virtual ~ThermalExpansionMaterial() = default;

  /**
   * @brief Set the reference-to-target transformation. This is required to use coefficient parameters.
   *
   * @param[in] Ttr The reference-to-target (stress-free) transformation
   */
  void setTransformation(mfem::ElementTransformation& Ttr) { parent_to_reference_transformation_ = &Ttr; }

  /**
   * @brief Evaluate the strain energy density function, W = W(F).
   *
   * @param[in] du_dX the displacement gradient
   * @return double Strain energy density
   */
  virtual void evalThermalDeformationGradient(mfem::DenseMatrix& dx_dX) = 0;

protected:
  /**
   * @brief Non-owning pointer to the reference element to stree-free configuration (target) transformation
   *
   */
  mfem::ElementTransformation* parent_to_reference_transformation_;
};

/**
 * @brief Neo-Hookean hyperelastic model with a strain energy density function given
 *   by the formula: \f$(\mu/2)(\bar{I}_1 - dim) + (K/2)(det(F)/g - 1)^2\f$ where
 *   F is the deformation gradient and \f$\bar{I}_1 = (det(F))^{-2/dim} Tr(F
 *   F^t)\f$. The parameters \f$\mu\f$ and \f$\lambda\f$ are the Lame parameters.
 *
 */
class IsotropicThermalExpansionMaterial : public ThermalExpansionMaterial {
public:
  IsotropicThermalExpansionMaterial(std::unique_ptr<mfem::Coefficient>&& coef_thermal_expansion,
                                    std::unique_ptr<mfem::Coefficient>&& reference_temp, FiniteElementState& temp)
      : c_coef_thermal_expansion_(std::move(coef_thermal_expansion)),
        c_reference_temp_(std::move(reference_temp)),
        temp_state_(temp)
  {
  }

  void evalThermalDeformationGradient(mfem::DenseMatrix& dx_dX);

  /**
   * @brief Destroy the Hyperelastic Material object
   *
   */
  virtual ~IsotropicThermalExpansionMaterial() = default;

  /**
   * @brief Disable the default constructor
   *
   */
  IsotropicThermalExpansionMaterial() = delete;

protected:
  /**
   * @brief Shear modulus in constant form
   *
   */
  mutable double coef_thermal_expansion_;

  /**
   * @brief Bulk modulus in constant form
   *
   */
  mutable double reference_temp_;

  mutable double temp_;

  /**
   * @brief Shear modulus in coefficient form
   *
   */
  std::unique_ptr<mfem::Coefficient> c_coef_thermal_expansion_;

  /**
   * @brief Bulk modulus in coefficient form
   *
   */
  std::unique_ptr<mfem::Coefficient> c_reference_temp_;

  /**
   * @brief The deformation gradient (dx_dX)
   *
   */
  FiniteElementState& temp_state_;

  /**
   * @brief Evaluate the coefficients
   * @note The reference-to-target transformation must be set before this call.
   *
   */
  inline void EvalCoeffs() const;
};

}  // namespace serac

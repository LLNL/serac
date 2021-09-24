// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermal_expansion_material.hpp
 *
 * @brief The thermal expansion material models for the solid module
 */

#pragma once

#include "mfem.hpp"
#include "serac/physics/utilities/finite_element_state.hpp"

namespace serac {

/**
 * @brief Abstract interface class for a generic thermal expansion model
 *
 */
class ThermalExpansionMaterial {
public:
  /**
   * @brief Construct a new thermal expansion material
   *
   */
  ThermalExpansionMaterial() : parent_to_reference_transformation_(nullptr) {}

  /**
   * @brief Destroy the thermal expansion material object
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
   * @brief Modify the displacement gradient to include thermal strain
   *
   * @param[inout] du_dX The unmodified displacement gradient
   * @pre The underlying element transformation must be set to the correct integration point
   * for the coefficient evaluations to work correctly
   */
  virtual void modifyDisplacementGradient(mfem::DenseMatrix& du_dX) = 0;

protected:
  /**
   * @brief Non-owning pointer to the reference element to stree-free configuration (target) transformation
   *
   */
  mfem::ElementTransformation* parent_to_reference_transformation_;
};

/**
 * @brief A simple isotropic finite deformation thermal expansion module.
 *
 * @details This implements the split deformation gradient model
 *
 * \f[
 * \mathbf{F} = \mathbf{F}_\theta \mathbf{F}_M
 * \f]
 *
 * where \f$\mathbf{F}\f$ is the total deformation gradient, \f$\mathbf{F}_\theta\f$ is the
 * deformation gradient induced by thermal expansion, and \f$\mathbf{F}_M\f$ is the mechanical
 * deformation gradient. The thermal deformation gradient of this model is given by
 *
 * \f[
 * \mathbf{F}_\theta = (1 + \alpha (\theta - \theta_\textrm{ref})) \mathbf{I}
 * \f]
 *
 * where \f$\alpha\f$ is the coefficient of thermal expansion, \f$\theta\f$ is the current
 * temperature, and \f$\theta_\textrm{ref}\f$ is the reference temperature.
 */
class IsotropicThermalExpansionMaterial : public ThermalExpansionMaterial {
public:
  /**
   * @brief Construct a new Isotropic Thermal Expansion Material object
   *
   * @param coef_thermal_expansion The coefficient of thermal expansion \f$\alpha\f$
   * @param reference_temp The reference temperature \f$\theta_{ref}\f$
   * @param temp The current temperature \f$\theta\f$
   */
  IsotropicThermalExpansionMaterial(std::unique_ptr<mfem::Coefficient>&& coef_thermal_expansion,
                                    std::unique_ptr<mfem::Coefficient>&& reference_temp, FiniteElementState& temp)
      : c_coef_thermal_expansion_(std::move(coef_thermal_expansion)),
        c_reference_temp_(std::move(reference_temp)),
        temp_state_(temp)
  {
  }

  /**
   * @brief Modify the displacement gradient to include thermal strain
   *
   * @param[inout] du_dX The unmodified displacement gradient
   * @pre The underlying element transformation must be set to the correct integration point
   * for the coefficient evaluations to work correctly
   */
  void modifyDisplacementGradient(mfem::DenseMatrix& du_dX);

  /**
   * @brief Destroy the isotropic thermal expansion object
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
   * @brief Coefficient of thermal expansion in constant form
   */
  mutable double coef_thermal_expansion_;

  /**
   * @brief Reference temperature in constant form
   */
  mutable double reference_temp_;

  /**
   * @brief Current temperature in constant form
   */
  mutable double temp_;

  /**
   * @brief Coefficient of thermal expansion in coefficient form
   */
  std::unique_ptr<mfem::Coefficient> c_coef_thermal_expansion_;

  /**
   * @brief Reference temperature in coefficient form
   *
   */
  std::unique_ptr<mfem::Coefficient> c_reference_temp_;

  /**
   * @brief Coefficient of thermal expansion in finite element state form
   */
  FiniteElementState& temp_state_;

  /**
   * @brief Evaluate the coefficients
   * @pre The reference-to-target transformation must be set before this call.
   *
   */
  inline void EvalCoeffs() const;
};

}  // namespace serac

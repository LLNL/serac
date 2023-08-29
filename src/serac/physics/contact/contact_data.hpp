// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file contact_data.hpp
 *
 * @brief Class for storing contact data
 */

#pragma once

#include <mfem/fem/pgridfunc.hpp>
#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/physics/contact/contact_config.hpp"
#include "serac/physics/state/finite_element_dual.hpp"
#include "serac/physics/state/finite_element_state.hpp"
#ifdef SERAC_USE_TRIBOL
#include "serac/physics/contact/contact_pair.hpp"
#endif

namespace serac {

namespace contact {

/**
 * @brief Default contact options: frictionless mortar with penalty = 1000
 * enforcement
 */
const ContactOptions default_contact_options = {.method      = ContactMethod::SingleMortar,
                                                .enforcement = ContactEnforcement::Penalty,
                                                .type        = ContactType::Frictionless,
                                                .penalty     = 1.0e3};

}  // namespace contact

/**
 * @brief This class stores contact pair data and interacts with Tribol
 */
class ContactData {
public:
  /**
   * @brief The constructor
   *
   * @param mesh The volume mesh for the problem
   */
  ContactData(const mfem::ParMesh& mesh);

  /**
   * @brief Destructor to finalize Tribol
   */
  ~ContactData();

  /**
   * @brief Add another contact pair
   *
   * @param pair_id Unique identifier for the ContactPair (used in Tribol)
   * @param bdry_attr_surf1 MFEM boundary attributes for the first surface
   * @param bdry_attr_surf2 MFEM boundary attributes for the second surface
   * @param contact_opts Defines contact method, enforcement, type, and penalty
   */
  void addContactPair(int pair_id, const std::set<int>& bdry_attr_surf1, const std::set<int>& bdry_attr_surf2,
                      ContactOptions contact_opts);

  /**
   * @brief Updates the positions, forces, and jacobian contributions associated with contact
   *
   * @param cycle The current simulation cycle
   * @param time The current time
   * @param dt The timestep size to attempt
   */
  void update(int cycle, double time, double& dt);

  /**
   * @brief Have there been contact pairs added?
   *
   * @return true if contact pairs have been added
   * @return false if there are no contact pairs
   */
  bool haveContactPairs() const;

  /**
   * @brief Get the contact constraint residual (i.e. nodal forces)
   *
   * @return Nodal contact forces as a Vector
   */
  FiniteElementDual forces() const;

  /**
   * @brief Returns pressures on the contact surfaces
   *
   * @return Pressure finite element state
   */
  mfem::Vector mergedPressures() const;

  /**
   * @brief Returns nodal gaps on true degrees of freedom
   *
   * @return Nodal gaps as a Vector
   */
  mfem::Vector mergedGaps() const;

  /**
   * @brief Returns a 2x2 block Jacobian on displacement/pressure degrees of
   * freedom from contact constraints
   *
   * @return Pointer to block Jacobian (2x2 BlockOperator of HypreParMatrix)
   */
  std::unique_ptr<mfem::BlockOperator> jacobian() const;

  /**
   * @brief Set the pressure field
   *
   * This sets Tribol's pressure degrees of freedom based on
   *  1) the values in true_pressure for Lagrange multiplier enforcement
   *  2) the nodal gaps and penalty for penalty enforcement
   *
   * @note The nodal gaps must be up-to-date for penalty enforcement
   *
   * @param merged_pressures Current pressure true dof values
   */
  void setPressures(const mfem::Vector& merged_pressures) const;

  /**
   * @brief Update the current coordinates based on the new displacement field
   *
   * @param u Current displacement dof values
   */
  void setDisplacements(const mfem::Vector& u);

  /**
   * @brief Are any contact pairs enforced using Lagrange multipliers?
   *
   * @return true: at least one contact pair is using Lagrange multiplier
   * enforcement
   * @return false: no contact pairs are using Lagrange multipliers
   */
  bool haveLagrangeMultipliers() const { return have_lagrange_multipliers_; }

  /**
   * @brief Get the number of Lagrange multiplier true degrees of freedom
   *
   * @return Number of Lagrange multiplier true degrees of freedom
   */
  int numPressureTrueDofs() const { return num_pressure_true_dofs_; };

  /**
   * @brief True degree of freedom offsets for each contact constraint
   *
   * @note Only includes constraints with Lagrange multiplier enforcement
   *
   * @return True degree of freedom offsets as an Array
   */
  mfem::Array<int> pressureTrueDofOffsets() const;

private:
#ifdef SERAC_USE_TRIBOL
  /**
   * @brief The volume mesh for the problem
   */
  const mfem::ParMesh& mesh_;
#endif

  /**
   * @brief Reference coordinates of the mesh
   */
  const mfem::ParGridFunction* reference_nodes_;

#ifdef SERAC_USE_TRIBOL
  /**
   * @brief Current coordinates of the mesh
   */
  mfem::ParGridFunction current_coords_;

  /**
   * @brief The contact boundary condition information
   */
  std::vector<ContactPair> pairs_;
#endif

  /**
   * @brief Offsets giving size of each Jacobian contribution
   */
  mutable mfem::Array<int> jacobian_offsets_;

  /**
   * @brief True if any of the contact pairs are enforced using Lagrange
   * multipliers
   */
  bool have_lagrange_multipliers_;

  /**
   * @brief Pressure T-dof count
   */
  int num_pressure_true_dofs_;
};

}  // namespace serac

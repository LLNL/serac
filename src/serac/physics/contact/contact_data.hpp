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

#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/physics/contact/contact_config.hpp"
#include "serac/physics/state/finite_element_dual.hpp"
#ifdef SERAC_USE_TRIBOL
#include "serac/physics/contact/contact_interaction.hpp"
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
 * @brief This class stores all ContactInteractions for a problem, calls Tribol functions that act on all contact
 * interactions, and agglomerates fields that exist over different ContactInteractions.
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
   * @brief Add another contact interaction
   *
   * @param interaction_id Unique identifier for the ContactInteraction (used in Tribol)
   * @param bdry_attr_surf1 MFEM boundary attributes for the first (mortar) surface
   * @param bdry_attr_surf2 MFEM boundary attributes for the second (nonmortar) surface
   * @param contact_opts Defines contact method, enforcement, type, and penalty
   */
  void addContactInteraction(int interaction_id, const std::set<int>& bdry_attr_surf1,
                             const std::set<int>& bdry_attr_surf2, ContactOptions contact_opts);

  /**
   * @brief Updates the positions, forces, and Jacobian contributions associated with contact
   *
   * @param cycle The current simulation cycle
   * @param time The current time
   * @param dt The timestep size to attempt
   */
  void update(int cycle, double time, double& dt);

  /**
   * @brief Get the contact constraint residual (i.e. nodal forces) from all contact interactions
   *
   * @return Nodal contact forces on the true DOFs
   */
  FiniteElementDual forces() const;

  /**
   * @brief Returns pressures from all contact interactions on the contact surface true degrees of freedom
   *
   * The type of pressure (normal or vector-valued) is set by the ContactType in the ContactOptions struct for the
   * contact interaction. TiedNormal and Frictionless (the two type supported in Tribol) correspond to scalar normal
   * pressure. Only linear (order = 1) pressure fields are supported.
   *
   * @return Pressure true degrees of freedom on each contact interaction (merged into one mfem::Vector)
   */
  mfem::Vector mergedPressures() const;

  /**
   * @brief Returns nodal gaps from all contact interactions on the contact surface true degrees of freedom
   *
   * The type of gap (normal or vector-valued) is set by the ContactType in the ContactOptions struct for the contact
   * interaction. TiedNormal and Frictionless (the two type supported in Tribol) correspond to scalar gap normal.  Only
   * linear (order = 1) gap fields are supported.
   *
   * @param [in] zero_inactive Sets inactive t-dofs to zero gap
   * @return Nodal gap true degrees of freedom on each contact interaction (merged into one mfem::Vector)
   */
  mfem::Vector mergedGaps(bool zero_inactive = false) const;

  /**
   * @brief Returns a 2x2 block Jacobian on displacement/pressure true degrees of
   * freedom from contact constraints
   *
   * The element Jacobian contributions are computed upon calling update(). This method does MPI communication to move
   * Jacobian contributions to the correct rank, then assembles the contributions.  The pressure degrees of freedom for
   * all contact interactions are merged into a single block.
   *
   * @note The blocks are owned by the BlockOperator
   *
   * @return Pointer to block Jacobian (2x2 BlockOperator of HypreParMatrix)
   */
  std::unique_ptr<mfem::BlockOperator> mergedJacobian() const;

  /**
   * @brief Computes the residual including contact terms
   *
   * @param [in] u Solution vector ([displacement; pressure] block vector)
   * @param [in,out] r Residual vector ([force; gap] block vector); takes in initialized residual force vector and adds
   * contact contributions
   */
  void residualFunction(const mfem::Vector& u, mfem::Vector& r);

  /**
   * @brief Computes the Jacobian including contact terms, given the non-contact Jacobian terms
   *
   * @param u Solution vector ([displacement; pressure] block vector)
   * @param orig_J The non-contact terms of the Jacobian, not including essential boundary conditions
   * @return Jacobian with contact terms, not including essential boundary conditions
   */
  std::unique_ptr<mfem::BlockOperator> jacobianFunction(const mfem::Vector& u, mfem::HypreParMatrix* orig_J) const;

  /**
   * @brief Set the pressure field
   *
   * This sets Tribol's pressure degrees of freedom based on
   *  1) the values in merged_pressure for Lagrange multiplier enforcement
   *  2) the nodal gaps and penalty for penalty enforcement
   *
   * @note The nodal gaps must be up-to-date for penalty enforcement
   *
   * @param merged_pressures Current pressure true dof values in a merged mfem::Vector
   */
  void setPressures(const mfem::Vector& merged_pressures) const;

  /**
   * @brief Update the current coordinates based on the new displacement field
   *
   * @param u Current displacement dof values
   */
  void setDisplacements(const mfem::Vector& u);

  /**
   * @brief Have there been contact interactions added?
   *
   * @return true if contact interactions have been added
   * @return false if there are no contact interactions
   */
  bool haveContactInteractions() const { return !interactions_.empty(); }

  /**
   * @brief Are any contact interactions enforced using Lagrange multipliers?
   *
   * @return true: at least one contact interaction is using Lagrange multiplier enforcement
   * @return false: no contact interactions are using Lagrange multipliers
   */
  bool haveLagrangeMultipliers() const { return have_lagrange_multipliers_; }

  /**
   * @brief Get the number of Lagrange multiplier true degrees of freedom
   *
   * @return Number of Lagrange multiplier true degrees of freedom
   */
  int numPressureDofs() const { return num_pressure_dofs_; };

  /**
   * @brief True degree of freedom offsets for each contact constraint
   *
   * @note Only includes constraints with Lagrange multiplier enforcement
   *
   * @return True degree of freedom offsets as an Array
   */
  mfem::Array<int> pressureDofOffsets() const;

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
  std::vector<ContactInteraction> interactions_;
#endif

  /**
   * @brief Offsets giving size of each Jacobian contribution
   */
  mutable mfem::Array<int> jacobian_offsets_;

  /**
   * @brief True if any of the contact interactions are enforced using Lagrange
   * multipliers
   */
  bool have_lagrange_multipliers_;

  /**
   * @brief Pressure T-dof count
   */
  int num_pressure_dofs_;
};

}  // namespace serac

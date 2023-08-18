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

#include <mfem/linalg/blockvector.hpp>
#include "mfem.hpp"

#include "serac/physics/contact/contact_pair.hpp"

namespace serac {

namespace contact {

const ContactOptions default_contact_options = {.method =      ContactMethod::SingleMortar,
                                                .enforcement = ContactEnforcement::Penalty,
                                                .type =        ContactType::Frictionless,
                                                .penalty =     1.0e3};

} // namespace contact

class ContactData {
public:

  /**
   * @brief The constructor
   *
   * @param mesh The volume mesh for the problem
   */
  ContactData(
    const mfem::ParMesh& mesh
  );

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
  void addContactPair(
    int pair_id,
    const std::set<int>& bdry_attr_surf1,
    const std::set<int>& bdry_attr_surf2,
    ContactOptions contact_opts
  );

  /**
   * @brief Updates the positions, forces, and jacobian contributions associated with contact
   *
   * @param cycle The current simulation cycle
   * @param time The current time
   * @param dt The timestep size to attempt
   * @param update_redecomp Re-builds redecomp mesh and updates data if true
   */
  void update(
    int cycle, 
    double time, 
    double& dt,
    bool update_redecomp = true
  );

  /**
   * @brief Return the contact data for each contact pair
   *
   * @return Vector of contact pairs
   */
  std::vector<ContactPair>& contactPairs() { return pairs_; }

  const std::vector<ContactPair>& contactPairs() const { return pairs_; }

  /**
   * @brief Get the abstract constraint residual
   *
   * @return The abstract constraint residual/RHS
   */
  mfem::Vector trueContactForces() const;

  mfem::Vector truePressures() const;

  mfem::Vector trueGaps() const;

  std::unique_ptr<mfem::BlockOperator> contactJacobian() const;

  /**
   * @brief Set the pressure field
   * 
   * @param true_pressures Current pressure true dof values
   */
  void setPressures(const mfem::Vector& true_pressures) const;

  /**
   * @brief Set the displacement field
   * 
   * @param true_displacements Current displacement true dof values
   */
  void setDisplacements(const mfem::Vector& true_displacements);

  /**
   * @brief Get the number of secondary nodes
   *
   * @return The number of secondary nodes
   */
  int numPressureTrueDofs() const { return num_pressure_true_dofs_; };

  mfem::Array<int> pressureTrueDofOffsets() const;

  /**
   * @brief Get the reference nodes
   *
   * @return Pointer to the reference nodes ParGridFunction
   */
  const mfem::ParGridFunction* referenceNodes() const { return reference_nodes_; };

private:

  /**
   * @brief The volume mesh for the problem
   */
  const mfem::ParMesh& mesh_;

  /**
   * @brief Reference coordinates of the mesh
   */
  const mfem::ParGridFunction* reference_nodes_;

  /**
   * @brief Current coordinates of the mesh
   */
  mfem::ParGridFunction current_coords_;

  /**
   * @brief The contact boundary condition information
   */
  std::vector<ContactPair> pairs_;

  /**
   * @brief Pressure T-dof count
   */
  int num_pressure_true_dofs_;

  /**
   * @brief Offsets giving size of each Jacobian contribution
   */
  mutable mfem::Array<int> jacobian_offsets_;
};

}  // namespace smith

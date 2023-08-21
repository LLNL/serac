// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file contact_pair.hpp
 *
 * @brief Class for storing a contact interaction and interfacing with Tribol
 */

#pragma once

#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/physics/contact/contact_config.hpp"
#include "serac/physics/state/finite_element_state.hpp"

#ifdef SERAC_USE_TRIBOL
#include "tribol/common/Parameters.hpp"
#endif

namespace serac {

/**
 * @brief Container for all interactions with the Tribol contact library
 *
 * This class provides a wrapper for all of the information needed by Tribol to
 * generate the mortar weights for the current penalty contact formulation. All
 * of the secondary nodes are currently considered active and therefore we are
 * enforcing an equality constraint.
 **/
class ContactPair {
public:

  /**
  * @brief The constructor
  *
  * @param pair_id Unique identifier for the ContactPair (used in Tribol)
  * @param mesh Mesh of the entire domain
  * @param bdry_attr_surf1 MFEM boundary attributes for the first surface
  * @param bdry_attr_surf2 MFEM boundary attributes for the second surface
  * @param current_coords Reference to the grid function holding current mesh
  * @param contact_opts Defines contact method, enforcement, type, and penalty
  * coordinates
  */
  ContactPair(
    int pair_id,
    const mfem::ParMesh& mesh,
    const std::set<int>& bdry_attr_surf1,
    const std::set<int>& bdry_attr_surf2,
    const mfem::ParGridFunction& current_coords,
    ContactOptions contact_opts
  );

  int getPairId() const { return pair_id_; }

  const ContactOptions& getContactOptions() const { return contact_opts_; }

  mfem::Vector contactForces() const;

  mfem::Vector gaps() const;

  mfem::ParGridFunction& pressure() const;

  /**
   * @brief Returns the number of pressure true DOFs on this rank.
   * 
   * @return int 
   */
  int numTruePressureDofs() const;

  /**
   * @brief Returns the total number of pressure true DOFs across all ranks.
   * 
   * @return int 
   */
  int numPressureDofs() const;

private:

#ifdef SERAC_USE_TRIBOL
  tribol::ContactMethod getMethod() const;
#endif

  /**
   * @brief Unique identifier for the contact interaction
   */
  int pair_id_;

  /**
   * @brief Defines contact method, enforcement, type, and penalty
   */
  ContactOptions contact_opts_;

  const mfem::ParGridFunction& current_coords_;

};

}  // namespace smith

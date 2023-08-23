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

#include "serac/serac_config.hpp"

#ifdef SERAC_USE_TRIBOL

#include "mfem.hpp"

#include "serac/physics/contact/contact_config.hpp"
#include "serac/physics/state/finite_element_state.hpp"

#include "tribol/common/Parameters.hpp"

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
  ContactPair(int pair_id, const mfem::ParMesh& mesh, const std::set<int>& bdry_attr_surf1,
              const std::set<int>& bdry_attr_surf2, const mfem::ParGridFunction& current_coords,
              ContactOptions contact_opts);

  /**
   * @brief Get the integer identifier of the contact pair
   *
   * @return Contact pair ID
   */
  int getPairId() const { return pair_id_; }

  /**
   * @brief Get the contact options for the contact pair
   *
   * @return Struct of contact options
   */
  const ContactOptions& getContactOptions() const { return contact_opts_; }

  /**
   * @brief Get the contact constraint residual for the contact pair
   *
   * @return Contact nodal forces as a Vector
   */
  mfem::Vector contactForces() const;

  /**
   * @brief Get the nodal gaps for the contact pair
   *
   * @return Nodal gaps as a Vector
   */
  mfem::Vector gaps() const;

  /**
   * @brief Get the pressure degrees of freedom for the contact pair
   *
   * @return Pressure degrees of freedom as a ParGridFunction
   */
  mfem::ParGridFunction& pressure() const;

  /**
   * @brief Returns the number of pressure true DOFs on this rank
   *
   * @return Number of pressure true DOFs as an integer
   */
  int numPressureTrueDofs() const;

  const mfem::Array<int>& inactiveTrueDofs() const;

private:
  /**
   * @brief Get the Tribol enforcement method given a serac enforcement method
   *
   * @return Tribol enforcement method
   */
  tribol::ContactMethod getMethod() const;

  /**
   * @brief Unique identifier for the contact interaction
   */
  int pair_id_;

  /**
   * @brief Defines contact method, enforcement, type, and penalty
   */
  ContactOptions contact_opts_;

  /**
   * @brief Reference to the current coords GridFunction
   */
  const mfem::ParGridFunction& current_coords_;

  /**
   * @brief List of true DOFs currently not in the active set
   */
  mutable mfem::Array<int> inactive_tdofs_;
};

}  // namespace serac

#endif

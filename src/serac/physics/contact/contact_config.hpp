// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file contact_config.hpp
 *
 * @brief This file contains enumerations and record types for contact configuration
 */

#pragma once

namespace serac {

/**
 * @brief Methodology for enforcing contact constraints (i.e. how you form the constraint equations)
 */
enum class ContactMethod
{
  SingleMortar /**< Puso and Laursen 2004 w/ approximate tangent */
};

/**
 * @brief Describes how to enforce the contact constraint equations
 */
enum class ContactEnforcement
{
  Penalty,           /**< Equal penalty applied to all constrained dofs */
  LagrangeMultiplier /**< Solve for exact pressures to satisfy constraints */
};

/**
 * @brief Mechanical constraint type on contact surfaces
 */
enum class ContactType
{
  TiedNormal,  /**< Tied contact in the normal direction, no friction */
  Frictionless /**< Enforce gap >= 0, pressure <= 0, gap * pressure = 0 in the normal direction */
};

/**
 * @brief Stores the options for a contact pair
 */
struct ContactOptions {
  /// The contact methodology to be applied
  ContactMethod method = ContactMethod::SingleMortar;

  /// The contact enforcement strategy to use
  ContactEnforcement enforcement = ContactEnforcement::Penalty;

  /// The type of contact constraint
  ContactType type = ContactType::Frictionless;

  /// Penalty parameter (only used when enforcement == ContactEnforcement::Penalty)
  double penalty = 1.0e3;
};

}  // namespace serac

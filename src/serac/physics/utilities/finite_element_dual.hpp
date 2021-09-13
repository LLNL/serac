// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file finite_element_dual.hpp
 *
 * @brief This contains a class that represents the dual of a finite element vector space, i.e. the space of residuals
 * and sensitivities.
 */

#pragma once

#include <memory>

#include "mfem.hpp"

#include "serac/physics/utilities/finite_element_vector.hpp"

namespace serac {

/**
 * @brief Class for encapsulating the dual vector space of a finite element space (i.e. the
 * space of linear forms as applied to a specific basis set)
 */
class FiniteElementDual : public FiniteElementVector {
public:
  /**
   * @brief Use the finite element vector constructors
   */
  using FiniteElementVector::FiniteElementVector;

  /**
   * @brief Returns a non-owning reference to the local degrees of freedom
   *
   * @return mfem::Vector& The local dof vector
   * @note While this is a grid function for plotting and parallelization, we only return a vector
   * type as the user should not use the interpolation capabilities of a grid function on the dual space
   * @note Shared degrees of freedom live on multiple MPI ranks
   */
  mfem::Vector& localVec() { return detail::retrieve(gf_); }

  /// @overload
  const mfem::Vector& localVec() const { return detail::retrieve(gf_); }

  /**
   * @brief Set a finite element dual to a constant value
   *
   * @param value The constant to set the finite element dual to
   * @return The modified finite element dual
   * @note This sets the true degrees of freedom and then broadcasts to the shared grid function entries. This means
   * that if a different value is given on different processors, a shared DOF will be set to the owning processor value.
   */
  FiniteElementDual& operator=(const double value)
  {
    FiniteElementVector::operator=(value);
    return *this;
  }
};

}  // namespace serac

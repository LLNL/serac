// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
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

#include "serac/physics/state/finite_element_vector.hpp"

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
  using FiniteElementVector::operator=;

  /**
   * @brief Fill a user-provided grid function based on the underlying true vector
   *
   * This distributes true vector dofs to the finite element (local) dofs  by multiplying the true dofs
   * by the restriction transpose operator.
   *
   * @see <a href="https://mfem.org/pri-dual-vec/">MFEM documentation</a> for details
   *
   */
  void fillGridFunction(mfem::ParGridFunction& grid_function) const
  {
    space_->GetRestrictionMatrix()->MultTranspose(*this, grid_function);
  }

  /**
   * @brief Initialize the true vector in the FiniteElementDual based on an input grid function
   *
   * This distributes the grid function dofs to the true vector dofs by multiplying by the
   * prolongation transpose operator.
   *
   * @see <a href="https://mfem.org/pri-dual-vec/">MFEM documentation</a> for details
   *
   * @param grid_function The grid function used to initialize the underlying true vector.
   */
  void setFromGridFunction(const mfem::ParGridFunction& grid_function) { grid_function.ParallelAssemble(*this); }
};

}  // namespace serac

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
  using FiniteElementVector::FiniteElementVector;
  using FiniteElementVector::operator=;
  using mfem::Vector::Print;

  /**
   * @brief Copy constructor
   *
   * @param[in] rhs The input Dual used for construction
   */
  FiniteElementDual(const FiniteElementDual& rhs) : FiniteElementVector(rhs) {}

  /**
   * @brief Move construct a new Finite Element Dual object
   *
   * @param[in] rhs The input vector used for construction
   */
  FiniteElementDual(FiniteElementDual&& rhs) : FiniteElementVector(std::move(rhs)) {}

  /**
   * @brief Copy assignment
   *
   * @param rhs The right hand side input Dual
   * @return The assigned FiniteElementDual
   */
  FiniteElementDual& operator=(const FiniteElementDual& rhs)
  {
    FiniteElementVector::operator=(rhs);
    return *this;
  }

  /**
   * @brief Fill a user-provided grid function based on the underlying true vector
   *
   * This distributes true vector dofs to the finite element (local) dofs  by multiplying the true dofs
   * by the restriction transpose operator.
   *
   * @see <a href="https://mfem.org/pri-dual-vec/">MFEM documentation</a> for details
   *
   * @note It is more mathematically correct to use a linear form on the dual space. However, grid functions
   * are common containers in MFEM-based codes, so this is left for convenience.
   *
   * @param grid_function The grid function to fill with the dual vector values
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
   * @note It is more mathematically correct to use a linear form on the dual space. However, grid functions
   * are common containers in MFEM-based codes, so this is left for convenience.
   *
   * @param grid_function The grid function used to initialize the underlying true vector.
   */
  void setFromGridFunction(const mfem::ParGridFunction& grid_function) { grid_function.ParallelAssemble(*this); }

  /**
   * @brief Fill a user-provided linear form based on the underlying true vector
   *
   * This distributes true vector dofs to the finite element (local) dofs by multiplying the true dofs
   * by the restriction transpose operator.
   *
   * @see <a href="https://mfem.org/pri-dual-vec/">MFEM documentation</a> for details
   *
   * @param linear_form The linear form used to initialize the underlying true vector.
   */
  void fillLinearForm(mfem::ParLinearForm& linear_form) const
  {
    space_->GetRestrictionMatrix()->MultTranspose(*this, linear_form);
  }

  /**
   * @brief Initialize the true vector in the FiniteElementDual based on an input linear form
   *
   * This distributes the linear form dofs to the true vector dofs by multiplying by the
   * prolongation transpose operator.
   *
   * @see <a href="https://mfem.org/pri-dual-vec/">MFEM documentation</a> for details
   *
   * @note The underlying MFEM call should really be const
   *
   * @param linear_form The linear form used to initialize the underlying true vector.
   */
  void setFromLinearForm(const mfem::ParLinearForm& linear_form)
  {
    const_cast<mfem::ParLinearForm&>(linear_form).ParallelAssemble(*this);
  }
};

}  // namespace serac

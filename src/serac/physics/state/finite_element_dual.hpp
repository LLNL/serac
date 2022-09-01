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
  using mfem::Vector::Print;
  /**
   * @brief Main constructor for building a new finite element Dual
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] options The options specified, namely those relating to the order of the problem,
   * the dimension of the FESpace, the type of FEColl, the DOF ordering that should be used,
   * and the name of the field
   */
  FiniteElementDual(
      mfem::ParMesh& mesh,
      Options&& options = {.order = 1, .vector_dim = 1, .coll = {}, .ordering = mfem::Ordering::byVDIM, .name = ""})
      : FiniteElementVector(mesh, std::move(options))
  {
  }

  /**
   * @brief Minimal constructor for a FiniteElementDual given a finite element space
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] space The space to use for the finite element Dual. This space is deep copied into the new FE Dual
   * @param[in] name The name of the field
   */
  FiniteElementDual(mfem::ParMesh& mesh, const mfem::ParFiniteElementSpace& space, const std::string& name = "")
      : FiniteElementVector(mesh, space, name)
  {
  }

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
   * @brief Copy assignment from a hypre par vector
   *
   * @param rhs The rhs input hypre par vector
   * @return The copy assigned Dual
   */
  FiniteElementDual& operator=(const mfem::HypreParVector& rhs)
  {
    FiniteElementVector::operator=(rhs);
    return *this;
  }

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

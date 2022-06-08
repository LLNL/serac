// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file boundary_condition.hpp
 *
 * @brief This file contains the declaration of the boundary condition class
 */

#pragma once

#include <memory>
#include <optional>
#include <set>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "serac/infrastructure/logger.hpp"
#include "serac/physics/state/finite_element_state.hpp"

namespace serac {

/**
 * @brief Boundary condition information bundle
 */
class BoundaryCondition {
public:
  /**
   * @brief Constructor for setting up a boundary condition using a set of attributes
   * @param[in] coef Either a mfem::Coefficient or mfem::VectorCoefficient representing the BC
   * @param[in] component The zero-indexed vector component if the BC applies to just one component,
   * should be -1 for all components
   * @param[in] attrs The set of boundary condition attributes in the mesh that the BC applies to
   * @param[in] num_attrs The total number of boundary attributes for the mesh
   * @param[in] state The finite element state on which this BC is applied
   */
  BoundaryCondition(GeneralCoefficient coef, const std::optional<int> component, const std::set<int>& attrs,
                    const int num_attrs = 0, FiniteElementState* state = nullptr);

  /**
   * @brief Minimal constructor for setting the true DOFs directly
   * @param[in] coef Either a mfem::Coefficient or mfem::VectorCoefficient representing the BC
   * @param[in] component The zero-indexed vector component if the BC applies to just one component,
   * should be -1 for all components
   * @param[in] true_dofs The indices of the relevant DOFs
   * @param[in] state The finite element state on which this BC is applied
   */
  BoundaryCondition(GeneralCoefficient coef, const std::optional<int> component, const mfem::Array<int>& true_dofs,
                    FiniteElementState* state = nullptr);

  /**
   * @brief Determines whether a boundary condition is associated with a tag
   * @tparam Tag The type of the tag to compare against
   * @param[in] tag The tag to compare against
   * @pre Template type "Tag" must be an enumeration
   */
  template <typename Tag>
  bool tagEquals(const Tag tag) const
  {
    static_assert(std::is_enum_v<Tag>, "Only enumerations can be used to tag a boundary condition.");
    SLIC_ERROR_ROOT_IF(!tag_, "No tag has been configured for this boundary condition");
    bool tags_same_type = typeid(tag).hash_code() == tag_->second;
    SLIC_WARNING_ROOT_IF(!tags_same_type, "Attempting to compare tags of two different enum types (always false)");
    return (static_cast<int>(tag) == tag_->first) && tags_same_type;
  }

  /**
   * @brief Sets the tag for the BC
   * @tparam Tag The template type for the tag (label)
   * @param[in] tag The new tag
   * @pre Template type "Tag" must be an enumeration
   */
  template <typename Tag>
  void setTag(const Tag tag)
  {
    static_assert(std::is_enum_v<Tag>, "Only enumerations can be used to tag a boundary condition.");
    tag_ = {static_cast<int>(tag), typeid(tag).hash_code()};
  }

  /**
   * @brief Returns a non-owning reference to the array of boundary
   * attribute markers
   */
  const mfem::Array<int>& markers() const { return markers_; }

  /**
   * @brief Returns a non-owning reference to the array of boundary
   * attribute markers
   */
  mfem::Array<int>& markers() { return markers_; }

  /**
   * @brief Accessor for the underlying vector coefficient
   *
   * This method performs an internal check to verify the underlying GeneralCoefficient
   * is in fact a vector.
   *
   * @return A non-owning reference to the underlying vector coefficient
   */
  const mfem::VectorCoefficient& vectorCoefficient() const;

  /**
   * @brief Accessor for the underlying vector coefficient
   *
   * This method performs an internal check to verify the underlying GeneralCoefficient
   * is in fact a vector.
   *
   * @return A non-owning reference to the underlying vector coefficient
   */
  mfem::VectorCoefficient& vectorCoefficient();

  /**
   * @brief Accessor for the underlying scalar coefficient
   *
   * This method performs an internal check to verify the underlying GeneralCoefficient
   * is in fact a scalar.
   *
   * @return A non-owning reference to the underlying scalar coefficient
   */
  const mfem::Coefficient& scalarCoefficient() const;

  /**
   * @brief Accessor for the underlying scalar coefficient
   *
   * This method performs an internal check to verify the underlying GeneralCoefficient
   * is in fact a scalar.
   *
   * @return A non-owning reference to the underlying scalar coefficient
   */
  mfem::Coefficient& scalarCoefficient();

  /**
   * @brief "Manually" set the DOF indices without specifying the field to which they apply
   * @param[in] true_dofs The true vector indices of the DOFs constrained by the boundary condition
   *
   * @note This will set both the true and local internal dof index arrays.
   * @note True and local dofs are described in the <a href="https://mfem.org/pri-dual-vec/">MFEM documentation</a>
   */
  void setTrueDofs(const mfem::Array<int> true_dofs);

  /**
   * @brief "Manually" set the DOF indices without specifying the field to which they apply
   * @param[in] local_dofs The local (finite element/grid function) indices of the DOFs constrained by the boundary
   * condition
   *
   * @note This will set both the true and local internal dof index arrays.
   * @note True and local dofs are described in the <a href="https://mfem.org/pri-dual-vec/">MFEM documentation</a>
   */
  void setLocalDofs(const mfem::Array<int> local_dofs);

  /**
   * @brief Returns the DOF indices for an essential boundary condition
   * @return A non-owning reference to the array of indices
   *
   * @note True and local dofs are described in the <a href="https://mfem.org/pri-dual-vec/">MFEM documentation</a>
   */
  const mfem::Array<int>& getTrueDofs() const
  {
    SLIC_ERROR_ROOT_IF(!true_dofs_, "True DOFs only available with essential BC.");
    return *true_dofs_;
  }

  /**
   * @brief Returns the DOF indices for an essential boundary condition
   * @return A non-owning reference to the array of indices
   *
   * @note True and local dofs are described in the <a href="https://mfem.org/pri-dual-vec/">MFEM documentation</a>
   */
  const mfem::Array<int>& getLocalDofs() const
  {
    SLIC_ERROR_ROOT_IF(!local_dofs_, "Local DOFs only available with essential BC.");
    return *local_dofs_;
  }

  /**
   * @brief Projects the boundary condition over a field
   * @param[inout] state The field to project over
   */
  void project(FiniteElementState& state) const;

  /**
   * @brief Projects the boundary condition over a grid function
   * @pre A corresponding field (FiniteElementState) has been associated
   * with the calling object via BoundaryCondition::setTrueDofs(FiniteElementState&)
   */
  void project() const;

  /**
   * @brief Projects the boundary condition over boundary DOFs of a grid function
   * @param[inout] gf The grid function representing the field to project over
   * @param[in] time The time for the coefficient, used for time-varying coefficients
   */
  void projectBdr(mfem::ParGridFunction& gf, const double time) const;

  /**
   * @brief Projects the boundary condition over boundary DOFs of a field
   * @param[inout] state The field to project over
   * @param[in] time The time for the coefficient, used for time-varying coefficients
   */
  void projectBdr(FiniteElementState& state, const double time) const;

  /**
   * @brief Projects the boundary condition over boundary DOFs
   * @param[in] time The time for the coefficient, used for time-varying coefficients
   * @pre A corresponding field (FiniteElementState) has been associated
   * with the calling object via BoundaryCondition::setTrueDofs(FiniteElementState&)
   */
  void projectBdr(const double time) const;

  /**
   * @brief Projects the boundary condition over boundary to a DoF vector
   * @param[in] dof_values The discrete dof values to project
   * @param[in] time The time for the coefficient, used for time-varying coefficients
   * @pre A corresponding field (FiniteElementState) has been associated
   * with the calling object via BoundaryCondition::setTrueDofs(FiniteElementState&)
   */
  void projectBdrToDofs(mfem::Vector& dof_values, const double time) const;

  /**
   * @brief Eliminates the rows and columns corresponding to the BC's true DOFS
   * from a stiffness matrix
   * @param[inout] k_mat The stiffness matrix to eliminate from,
   * will be modified.  These eliminated matrix entries can be
   * used to eliminate an essential BC to an RHS vector with
   * BoundaryCondition::eliminateToRHS
   */
  void eliminateFromMatrix(mfem::HypreParMatrix& k_mat) const;

  /**
   * @brief Eliminates boundary condition from solution to RHS
   * @param[in] k_mat_post_elim A stiffness matrix post-elimination
   * @param[in] soln The solution vector
   * @param[out] rhs The RHS vector for the system
   * @pre BoundaryCondition::eliminateFrom has been called
   */
  void eliminateToRHS(mfem::HypreParMatrix& k_mat_post_elim, const mfem::Vector& soln, mfem::Vector& rhs) const;

  /**
   * @brief Applies an essential boundary condition to RHS
   * @param[in] k_mat_post_elim A stiffness (system) matrix post-elimination
   * @param[out] rhs The RHS vector for the system
   * @param[inout] state The state from which the solution DOF values are extracted and used to eliminate
   * @param[in] time Simulation time, used for time-varying boundary coefficients
   * @pre BoundaryCondition::eliminateFrom has been called
   */
  void apply(mfem::HypreParMatrix& k_mat_post_elim, mfem::Vector& rhs, FiniteElementState& state,
             const double time = 0.0) const;

  /**
   * @brief Sets the underlying coefficient's time
   *
   * @param[in] time The current simulation time
   *
   * Used for time-dependent coefficients
   */
  void setTime(const double time);

private:
  /**
   * @brief Uses mfem::ParFiniteElementSpace::GetEssentialTrueDofs to
   * determine the DOFs for the boundary condition from the stored marker list
   *
   * @note This will set both the true and local dof values.
   */
  void setDofs();

  /**
   * @brief A coefficient containing either a mfem::Coefficient or an mfem::VectorCoefficient
   */
  GeneralCoefficient coef_;
  /**
   * @brief The vector component affected by this BC (empty implies all components)
   */
  std::optional<int> component_;
  /**
   * @brief The attribute marker array where this BC is active
   */
  mfem::Array<int> markers_;
  /**
   * @brief The true DOFs affected by this BC
   * @note Only used for essential (Dirichlet) BCs
   */
  std::optional<mfem::Array<int>> true_dofs_;
  /**
   * @brief The local (finite element) DOFs affected by this BC
   * @note Only used for essential (Dirichlet) BCs
   */
  std::optional<mfem::Array<int>> local_dofs_;
  /**
   * @brief The state (field) affected by this BC
   * @note Only used for essential (Dirichlet) BCs
   */
  FiniteElementState* state_ = nullptr;
  /**
   * @brief The eliminated entries for Dirichlet BCs
   */
  mutable std::unique_ptr<mfem::HypreParMatrix> eliminated_matrix_entries_;
  /**
   * @brief A label for the BC, for filtering purposes, in addition to its type hash
   * @note This should always correspond to an enum
   * The first element is the enum val, the second is the hash of the corresponding enum type
   */
  std::optional<std::pair<int, std::size_t>> tag_;
};

}  // namespace serac

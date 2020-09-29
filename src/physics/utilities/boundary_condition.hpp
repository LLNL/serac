// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file boundary_condition.hpp
 *
 * @brief This file contains the declaration of the boundary condition class
 */

#ifndef BOUNDARY_CONDITION
#define BOUNDARY_CONDITION

#include <memory>
#include <optional>
#include <set>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <variant>

#include "infrastructure/logger.hpp"
#include "physics/utilities/finite_element_state.hpp"
#include "physics/utilities/solver_config.hpp"

namespace serac {

class GeneralCoefficientWrapper {
public:
  GeneralCoefficientWrapper(GeneralCoefficient coef) : coef_(coef) {}

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

private:
  /**
   * @brief A coefficient containing either a mfem::Coefficient or an mfem::VectorCoefficient
   */
  GeneralCoefficient coef_;
};

class EssentialBoundaryCondition {
public:
  /**
   * @brief Constructor for setting up a boundary condition using a set of attributes
   * @param[in] coef Either a mfem::Coefficient or mfem::VectorCoefficient representing the BC
   * @param[in] component The zero-indexed vector component if the BC applies to just one component,
   * should be -1 for all components
   * @param[in] markers The 0-1 array of boundary condition markers
   */
  EssentialBoundaryCondition(GeneralCoefficient coef, const int component, mfem::Array<int>&& markers);

  /**
   * @brief Constructor for setting up a boundary condition using already-known DOFS (to be set with setTrueDofs)
   * @param[in] coef Either a mfem::Coefficient or mfem::VectorCoefficient representing the BC
   * @param[in] component The zero-indexed vector component if the BC applies to just one component,
   * should be -1 for all components
   */
  EssentialBoundaryCondition(GeneralCoefficient coef, const int component);

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
   * @brief "Manually" set the DOF indices without specifying the field to which they apply
   * @param[in] dofs The indices of the DOFs constrained by the boundary condition
   */
  void setTrueDofs(const mfem::Array<int>& dofs);

  /**
   * @brief Uses mfem::ParFiniteElementSpace::GetEssentialTrueDofs to
   * determine the DOFs for the boundary condition
   * @param[in] state The finite element state for which the DOFs should be obtained
   */
  void setTrueDofs(FiniteElementState& state);

  /**
   * @brief Returns the DOF indices for an essential boundary condition
   * @return A non-owning reference to the array of indices
   */
  const mfem::Array<int>& getTrueDofs() const { return true_dofs_; }

  /**
   * @brief Projects the boundary condition over a field
   * @param[inout] state The field to project over
   */
  void project(FiniteElementState& state) const;

  /**
   * @brief Projects the boundary condition over boundary DOFs of a grid function
   * @param[inout] gf The grid function representing the field to project over
   * @param[in] time The time for the coefficient, used for time-varying coefficients
   * @param[in] should_be_scalar Whether the boundary condition coefficient should be a scalar coef
   */
  void projectBdr(mfem::ParGridFunction& gf, const double time, const bool should_be_scalar = true) const;

  /**
   * @brief Projects the boundary condition over boundary DOFs of a field
   * @param[inout] state The field to project over
   * @param[in] time The time for the coefficient, used for time-varying coefficients
   * @param[in] should_be_scalar Whether the boundary condition coefficient should be a scalar coef
   */
  void projectBdr(FiniteElementState& state, const double time, const bool should_be_scalar = true) const;

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
   * @param[in] should_be_scalar Whether the boundary coefficient should be a scalar coefficient
   * @pre BoundaryCondition::eliminateFrom has been called
   */
  void apply(mfem::HypreParMatrix& k_mat_post_elim, mfem::Vector& rhs, FiniteElementState& state,
             const double time = 0.0, const bool should_be_scalar = true) const;

private:
  /**
   * @brief A coefficient containing either a mfem::Coefficient or an mfem::VectorCoefficient
   * @note Marked mutable as changing things like the time associated with the coef doesn't
   * really affect the state of the BoundaryCondition object
   */
  mutable GeneralCoefficientWrapper coef_;
  /**
   * @brief The vector component affected by this BC (-1 implies all components)
   */
  int component_;
  /**
   * @brief The attribute marker array where this BC is active
   */
  mfem::Array<int> markers_;
  /**
   * @brief The true DOFs affected by this BC
   */
  mfem::Array<int> true_dofs_;
  /**
   * @brief Whether the DOFs have been fully initialized
   */
  bool dofs_fully_initialized_;
  /**
   * @brief The eliminated entries for Dirichlet BCs
   */
  mutable std::unique_ptr<mfem::HypreParMatrix> eliminated_matrix_entries_;
};

class NaturalBoundaryCondition {
public:
  /**
   * @brief Constructor for setting up a boundary condition using a set of attributes
   * @param[in] coef Either a mfem::Coefficient or mfem::VectorCoefficient representing the BC
   * @param[in] component The zero-indexed vector component if the BC applies to just one component,
   * should be -1 for all components
   * @param[in] markers The 0-1 array of boundary condition markers
   */
  NaturalBoundaryCondition(GeneralCoefficient coef, const int component, mfem::Array<int>&& markers);

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
   * @see GeneralCoefficientWrapper
   */
  const mfem::VectorCoefficient& vectorCoefficient() const { return coef_.vectorCoefficient(); }

  /**
   * @brief Accessor for the underlying vector coefficient
   * @see GeneralCoefficientWrapper
   */
  mfem::VectorCoefficient& vectorCoefficient() { return coef_.vectorCoefficient(); }

  /**
   * @brief Accessor for the underlying scalar coefficient
   * @see GeneralCoefficientWrapper
   */
  const mfem::Coefficient& scalarCoefficient() const { return coef_.scalarCoefficient(); }

  /**
   * @brief Accessor for the underlying scalar coefficient
   * @see GeneralCoefficientWrapper
   */
  mfem::Coefficient& scalarCoefficient() { return coef_.scalarCoefficient(); }

private:
  /**
   * @brief A coefficient containing either a mfem::Coefficient or an mfem::VectorCoefficient
   */
  GeneralCoefficientWrapper coef_;
  /**
   * @brief The vector component affected by this BC (-1 implies all components)
   */
  int component_;
  /**
   * @brief The attribute marker array where this BC is active
   */
  mfem::Array<int> markers_;
};

}  // namespace serac

#endif

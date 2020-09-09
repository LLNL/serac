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
#include <variant>

#include "common/finite_element_state.hpp"
#include "common/serac_types.hpp"

namespace serac {

/**
 * @brief Boundary condition information bundle
 */
class BoundaryCondition {
public:
  /**
   * Constructor for setting up a boundary condition using a set of attributes
   * @param[in] coef Either a mfem::Coefficient or mfem::VectorCoefficient representing the BC
   * @param[in] component The zero-indexed vector component if the BC applies to just one component,
   * should be -1 for all components
   * @param[in] attrs The set of boundary condition attributes in the mesh that the BC applies to
   * @param[in] num_attrs The total number of boundary attributes for the mesh
   */
  BoundaryCondition(GeneralCoefficient coef, const int component, const std::set<int>& attrs, const int num_attrs = 0);

  /**
   * Minimal constructor for setting the true DOFs directly
   * @param[in] coef Either a mfem::Coefficient or mfem::VectorCoefficient representing the BC
   * @param[in] component The zero-indexed vector component if the BC applies to just one component,
   * should be -1 for all components
   * @param[in] true_dofs The indices of the relevant DOFs
   */
  BoundaryCondition(GeneralCoefficient coef, const int component, const mfem::Array<int>& true_dofs);

  const mfem::Array<int>& markers() const { return markers_; }

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
   * "Manually" set the DOF indices without specifying the field to which they apply
   * @param[in] dofs The indices of the DOFs constrained by the boundary condition
   */
  void setTrueDofs(const mfem::Array<int> dofs);

  /**
   * Uses mfem::ParFiniteElementSpace::GetEssentialTrueDofs to
   * determine the DOFs for the boundary condition
   * @param[in] state The finite element state for which the DOFs should be obtained
   */
  void setTrueDofs(FiniteElementState& state);

  const mfem::Array<int>& getTrueDofs() const
  {
    SLIC_ERROR_IF(!true_dofs_, "True DOFs only available with essential BC.");
    return *true_dofs_;
  }

  /**
   * Projects the boundary condition over a field
   * @param[inout] state The field to project over
   * @param[in] fes The finite element space that should be used to generate
   * the scalar DOF list
   */
  void project(FiniteElementState& state) const;

  /**
   * Projects the boundary condition over a grid function
   * @pre A corresponding field (FiniteElementState) has been associated
   * with the calling object via BoundaryCondition::setTrueDofs(FiniteElementState&)
   */
  void project() const;

  /**
   * Projects the boundary condition over boundary DOFs of a grid function
   * @param[inout] gf The grid function representing the field to project over
   * @param[in] time The time for the coefficient, used for time-varying coefficients
   * @param[in] should_be_scalar Whether the boundary condition coefficient should be a scalar coef
   */
  void projectBdr(mfem::ParGridFunction& gf, const double time, const bool should_be_scalar = true) const;

  /**
   * Projects the boundary condition over boundary DOFs of a field
   * @param[inout] state The field to project over
   * @param[in] time The time for the coefficient, used for time-varying coefficients
   * @param[in] should_be_scalar Whether the boundary condition coefficient should be a scalar coef
   */
  void projectBdr(FiniteElementState& state, const double time, const bool should_be_scalar = true) const;

  /**
   * Projects the boundary condition over boundary DOFs
   * @param[in] time The time for the coefficient, used for time-varying coefficients
   * @param[in] should_be_scalar Whether the boundary condition coefficient should be a scalar coef
   * @pre A corresponding field (FiniteElementState) has been associated
   * with the calling object via BoundaryCondition::setTrueDofs(FiniteElementState&)
   */
  void projectBdr(const double time, const bool should_be_scalar = true) const;

  /**
   * Allocates an integrator of type "Integrator" on the heap,
   * constructing it with the boundary condition's vector coefficient,
   * intended to be passed to mfem::*LinearForm::Add*Integrator
   * @return An owning pointer to the new integrator
   * @pre Requires Integrator::Integrator(mfem::VectorCoefficient&)
   */
  template <typename Integrator>
  std::unique_ptr<Integrator> newVecIntegrator() const;

  /**
   * Allocates an integrator of type "Integrator" on the heap,
   * constructing it with the boundary condition's coefficient,
   * intended to be passed to mfem::*LinearForm::Add*Integrator
   * @return An owning pointer to the new integrator
   * @pre Requires Integrator::Integrator(mfem::Coefficient&)
   */
  template <typename Integrator>
  std::unique_ptr<Integrator> newIntegrator() const;

  /**
   * Eliminates the rows and columns corresponding to the BC's true DOFS
   * from a stiffness matrix
   * @param[inout] k_mat The stiffness matrix to eliminate from,
   * will be modified.  These eliminated matrix entries can be
   * used to eliminate an essential BC to an RHS vector with
   * BoundaryCondition::eliminateToRHS
   */
  void eliminateFromMatrix(mfem::HypreParMatrix& k_mat) const;

  /**
   * Eliminates boundary condition from solution to RHS
   * @param[in] k_mat_post_elim A stiffness matrix post-elimination
   * @param[in] soln The solution vector
   * @param[out] rhs The RHS vector for the system
   * @pre BoundaryCondition::eliminateFrom has been called
   */
  void eliminateToRHS(mfem::HypreParMatrix& k_mat_post_elim, const mfem::Vector& soln, mfem::Vector& rhs) const;

  /**
   * Applies an essential boundary condition to RHS
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
   */
  GeneralCoefficient coef_;
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
   * @note Only used for essential (Dirichlet) BCs
   */
  std::optional<mfem::Array<int>> true_dofs_;
  /**
   * @brief The state (field) affected by this BC
   * @note Only used for essential (Dirichlet) BCs
   */
  FiniteElementState* state_ = nullptr;
  /**
   * @brief The eliminated entries for Dirichlet BCs
   */
  mutable std::unique_ptr<mfem::HypreParMatrix> eliminated_matrix_entries_;
};

}  // namespace serac

#endif

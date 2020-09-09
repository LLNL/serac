// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file boundary_condition_manager.hpp
 *
 * @brief This file contains the declaration of the boundary condition manager class
 */

#ifndef BOUNDARY_CONDITION_MANAGER
#define BOUNDARY_CONDITION_MANAGER

#include <memory>
#include <set>

#include "common/boundary_condition.hpp"
#include "common/finite_element_state.hpp"
#include "common/serac_types.hpp"

namespace serac {

class BoundaryConditionManager {
public:
  /**
   * @brief Set the essential boundary conditions from a list of boundary markers and a coefficient
   *
   * @param[in] ess_bdr The set of essential BC attributes
   * @param[in] ess_bdr_coef The essential BC value coefficient
   * @param[in] state The finite element state to which the BC should be applied
   * @param[in] component The component to set (-1 implies all components are set)
   */
  void addEssential(const std::set<int>& ess_bdr, serac::GeneralCoefficient ess_bdr_coef, FiniteElementState& state,
                    const int component = -1);

  /**
   * @brief Set the natural boundary conditions from a list of boundary markers and a coefficient
   *
   * @param[in] nat_bdr The set of mesh attributes denoting a natural boundary
   * @param[in] nat_bdr_coef The coefficient defining the natural boundary function
   * @param[in] state The finite element state to which the BC should be applied
   * @param[in] component The component to set (-1 implies all components are set)
   */
  void addNatural(const std::set<int>& nat_bdr, serac::GeneralCoefficient nat_bdr_coef, FiniteElementState& state,
                  const int component = -1);

  /**
   * @brief Set a generic boundary condition from a list of boundary markers and a coefficient
   *
   * @param[in] bdr_attr The set of mesh attributes denoting a natural boundary
   * @param[in] bdr_coef The coefficient defining the natural boundary function
   * @param[in] state The finite element state to which the BC should be applied
   * @param[in] component The component to set (-1 implies all components are set)
   */
  void addGeneric(const std::set<int>& bdr_attr, serac::GeneralCoefficient bdr_coef, FiniteElementState& state,
                  const int component = -1);

  /**
   * @brief Set a list of true degrees of freedom from a coefficient
   *
   * @param[in] true_dofs The true degrees of freedom to set with a Dirichlet condition
   * @param[in] ess_bdr_coef The coefficient that evaluates to the Dirichlet condition
   * @param[in] component The component to set (-1 implies all components are set)
   */
  void addTrueDofs(const mfem::Array<int>& true_dofs, serac::GeneralCoefficient ess_bdr_coef, int component = -1);

  /**
   * @brief Returns all the degrees of freedom associated with all the essential BCs
   * @return A const reference to the list of DOF indices, without duplicates and sorted
   */
  const mfem::Array<int>& allDofs() const
  {
    if (!all_dofs_valid_) {
      updateAllDofs();
    }
    return all_dofs_;
  }

  /**
   * @brief Eliminates all essential BCs from a matrix
   * @param[inout] matrix The matrix to eliminate from, will be modified
   * @return The eliminated matrix entries
   * @note The sum of the eliminated matrix and the modified parameter is
   * equal to the initial state of the parameter
   */
  std::unique_ptr<mfem::HypreParMatrix> eliminateAllFromMatrix(mfem::HypreParMatrix& matrix) const
  {
    return std::unique_ptr<mfem::HypreParMatrix>(matrix.EliminateRowsCols(allDofs()));
  }

  /**
   * @brief Accessor for the essential BC objects
   */
  std::vector<BoundaryCondition>& essentials() { return ess_bdr_; }
  /**
   * @brief Accessor for the natural BC objects
   */
  std::vector<BoundaryCondition>& naturals() { return nat_bdr_; }
  /**
   * @brief Accessor for the generic BC objects
   */
  std::vector<BoundaryCondition>& generics() { return other_bdr_; }

  /**
   * @brief Accessor for the essential BC objects
   */
  const std::vector<BoundaryCondition>& essentials() const { return ess_bdr_; }
  /**
   * @brief Accessor for the natural BC objects
   */
  const std::vector<BoundaryCondition>& naturals() const { return nat_bdr_; }
  /**
   * @brief Accessor for the generic BC objects
   */
  const std::vector<BoundaryCondition>& generics() const { return other_bdr_; }

private:
  /**
   * @brief Updates the "cached" list of all DOF indices
   */
  void updateAllDofs() const;

  /**
   * @brief The vector of essential boundary conditions
   */
  std::vector<BoundaryCondition> ess_bdr_;

  /**
   * @brief The vector of natural boundary conditions
   */
  std::vector<BoundaryCondition> nat_bdr_;

  /**
   * @brief The vector of generic (not Dirichlet or Neumann) boundary conditions
   */
  std::vector<BoundaryCondition> other_bdr_;

  /**
   * @brief The set of boundary attributes associated with
   * already-registered BCs
   * @see https://mfem.org/mesh-formats/
   */
  std::set<int> attrs_in_use_;

  /**
   * @brief The set of true DOF indices corresponding
   * to all registered BCs
   */
  mutable mfem::Array<int> all_dofs_;

  /**
   * @brief Whether the set of stored total DOFs is valid
   */
  mutable bool all_dofs_valid_ = false;
};

}  // namespace serac

#endif

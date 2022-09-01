// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file finite_element_state.hpp
 *
 * @brief This file contains the declaration of structure that manages the MFEM objects
 * that make up the state for a given field
 */

#pragma once

#include <functional>
#include <memory>
#include <optional>

#include "mfem.hpp"

#include "serac/infrastructure/variant.hpp"
#include "serac/physics/state/finite_element_vector.hpp"

namespace serac {

/**
 * @brief convenience function for querying the type stored in a GeneralCoefficient
 */
inline bool is_scalar_valued(const GeneralCoefficient& coef)
{
  return holds_alternative<std::shared_ptr<mfem::Coefficient>>(coef);
}

/**
 * @brief convenience function for querying the type stored in a GeneralCoefficient
 */
inline bool is_vector_valued(const GeneralCoefficient& coef)
{
  return holds_alternative<std::shared_ptr<mfem::VectorCoefficient>>(coef);
}

/**
 * @brief Class for encapsulating the critical MFEM components of a primal finite element field
 *
 * Namely: Mesh, FiniteElementCollection, FiniteElementState, and the true vector of the solution
 */
class FiniteElementState : public FiniteElementVector {
public:
  using mfem::Vector::Print;

  /**
   * @brief Main constructor for building a new finite element state
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] options The options specified, namely those relating to the order of the problem,
   * the dimension of the FESpace, the type of FEColl, the DOF ordering that should be used,
   * and the name of the field
   */
  FiniteElementState(
      mfem::ParMesh& mesh,
      Options&& options = {.order = 1, .vector_dim = 1, .coll = {}, .ordering = mfem::Ordering::byVDIM, .name = ""})
      : FiniteElementVector(mesh, std::move(options))
  {
  }

  /**
   * @brief Minimal constructor for a FiniteElementState given a finite element space
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] space The space to use for the finite element state. This space is deep copied into the new FE state
   * @param[in] name The name of the field
   */
  FiniteElementState(mfem::ParMesh& mesh, const mfem::ParFiniteElementSpace& space, const std::string& name = "")
      : FiniteElementVector(mesh, space, name)
  {
  }

  /**
   * @brief Copy constructor
   *
   * @param[in] rhs The input state used for construction
   */
  FiniteElementState(const FiniteElementState& rhs) : FiniteElementVector(rhs) {}

  /**
   * @brief Move construct a new Finite Element State object
   *
   * @param[in] rhs The input vector used for construction
   */
  FiniteElementState(FiniteElementState&& rhs) : FiniteElementVector(std::move(rhs)) {}

  /**
   * @brief Copy assignment
   *
   * @param rhs The right hand side input state
   * @return The assigned FiniteElementState
   */
  FiniteElementState& operator=(const FiniteElementState& rhs)
  {
    FiniteElementVector::operator=(rhs);
    return *this;
  }

  /**
   * @brief Copy assignment from a hypre par vector
   *
   * @param rhs The rhs input hypre par vector
   * @return The copy assigned state
   */
  FiniteElementState& operator=(const mfem::HypreParVector& rhs)
  {
    FiniteElementVector::operator=(rhs);
    return *this;
  }

  /**
   * @brief Set a finite element state to a constant value
   *
   * @param value The constant to set the finite element state to
   * @return The modified finite element state
   * @note This sets the true degrees of freedom and then broadcasts to the shared grid function entries. This means
   * that if a different value is given on different processors, a shared DOF will be set to the owning processor value.
   */
  FiniteElementState& operator=(const double value)
  {
    FiniteElementVector::operator=(value);
    return *this;
  }

  /**
   * @brief Fill a user-provided grid function based on the underlying true vector
   *
   * This distributes true vector dofs to the finite element (local) dofs  by multiplying the true dofs
   * by the prolongation operator.
   *
   * @see <a href="https://mfem.org/pri-dual-vec/">MFEM documentation</a> for details
   *
   */
  void fillGridFunction(mfem::ParGridFunction& grid_function) const { grid_function.SetFromTrueDofs(*this); }

  /**
   * @brief Initialize the true vector in the FiniteElementState based on an input grid function
   *
   * This distributes the grid function dofs to the true vector dofs by multiplying by the
   * restriction operator.
   *
   * @see <a href="https://mfem.org/pri-dual-vec/">MFEM documentation</a> for details
   *
   * @param grid_function The grid function used to initialize the underlying true vector.
   */
  void setFromGridFunction(const mfem::ParGridFunction& grid_function) { grid_function.GetTrueDofs(*this); }
};

}  // namespace serac

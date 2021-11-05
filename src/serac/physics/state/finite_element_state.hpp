// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
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
 * @brief A sum type for encapsulating either a scalar or vector coeffient
 */
using GeneralCoefficient = variant<std::shared_ptr<mfem::Coefficient>, std::shared_ptr<mfem::VectorCoefficient>>;

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
 * Namely: Mesh, FiniteElementCollection, FiniteElementState,
 * GridFunction, and a Vector of the solution
 */
class FiniteElementState : public FiniteElementVector {
public:
  /**
   * @brief Use the finite element vector constructors
   */
  using FiniteElementVector::FiniteElementVector;

  /**
   * @brief Set the internal grid function using the true DOF values
   *
   * This distributes true vector dofs to the finite element (local) dofs  by multiplying the true dofs
   * by the prolongation operator.
   *
   * @see <a href="https://mfem.org/pri-dual-vec/">MFEM documentation</a> for details
   *
   */
  void distributeSharedDofs() { detail::retrieve(gf_).SetFromTrueDofs(true_vec_); }

  /**
   * @brief Initialize the true vector from the grid function values
   *
   * This initializes the true vector dofs by multiplying the finite element dofs
   * by the restriction operator.
   *
   * @see <a href="https://mfem.org/pri-dual-vec/">MFEM documentation</a> for details
   */
  void initializeTrueVec() { detail::retrieve(gf_).GetTrueDofs(true_vec_); }

  /**
   * Returns a non-owning reference to the internal grid function
   */
  mfem::ParGridFunction& gridFunc() { return detail::retrieve(gf_); }
  /// \overload
  const mfem::ParGridFunction& gridFunc() const { return detail::retrieve(gf_); }

  /**
   * Returns a GridFunctionCoefficient referencing the internal grid function
   */
  mfem::GridFunctionCoefficient gridFuncCoef() const
  {
    const auto& gf = detail::retrieve(gf_);
    return mfem::GridFunctionCoefficient{&gf, gf.VectorDim()};
  }

  /**
   * Returns a VectorGridFunctionCoefficient referencing the internal grid function
   */
  mfem::VectorGridFunctionCoefficient vectorGridFuncCoef() const
  {
    return mfem::VectorGridFunctionCoefficient{&detail::retrieve(gf_)};
  }

  /**
   * Projects a coefficient (vector or scalar) onto the field
   * @param[in] coef The coefficient to project
   */
  void project(const GeneralCoefficient& coef)
  {
    // The generic lambda parameter, auto&&, allows the component type (mfem::Coef or mfem::VecCoef)
    // to be deduced, and the appropriate version of ProjectCoefficient is dispatched.
    visit(
        [this](auto&& concrete_coef) {
          detail::retrieve(gf_).ProjectCoefficient(*concrete_coef);
          initializeTrueVec();
        },
        coef);
  }
  /// \overload
  void project(mfem::Coefficient& coef)
  {
    detail::retrieve(gf_).ProjectCoefficient(coef);
    initializeTrueVec();
  }
  /// \overload
  void project(mfem::VectorCoefficient& coef)
  {
    detail::retrieve(gf_).ProjectCoefficient(coef);
    initializeTrueVec();
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
};

/**
 * @brief Calculate the Lp norm of a finite element state
 *
 * @param state The state variable to compute a norm of
 * @param p Order of the norm
 * @return The norm value
 */
double norm(const FiniteElementState& state, double p = 2);

}  // namespace serac

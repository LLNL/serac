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
   * Main constructor for building a new finite element dual
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] options The options specified, namely those relating to the order of the problem,
   * the dimension of the FESpace, the type of FEColl, the DOF ordering that should be used,
   * and the name of the field
   */
  FiniteElementDual(mfem::ParMesh& mesh, Options&& options = {})
      : FiniteElementVector(mesh, std::forward<FiniteElementVector::Options>(options)){};

  /**
   * @brief Minimal constructor for a FiniteElementDual given an already-existing field
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] gf The field for the dual to create (object does not take ownership)
   * @param[in] name The name of the field
   */
  FiniteElementDual(mfem::ParMesh& mesh, mfem::ParGridFunction& gf, const std::string& name = "")
      : FiniteElementVector(mesh, gf, name){};

  /**
   * @brief Minimal constructor for a FiniteElementDual given a finite element space
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] space The space to use for the finite element dual. This space is deep copied into the new FE state
   * @param[in] name The name of the field
   */
  FiniteElementDual(mfem::ParMesh& mesh, mfem::ParFiniteElementSpace& space, const std::string& name = "")
      : FiniteElementVector(mesh, space, name){};

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

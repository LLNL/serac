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

#include "serac/physics/utilities/finite_element_state.hpp"

namespace serac {

/**
 * @brief Class for encapsulating the dual vector space of a finite element space (i.e. the
 * space of linear forms as applied to a specific basis set)
 */
class FiniteElementDual {
public:
  /**
   * @brief Structure for optionally configuring a FiniteElementDual
   */
  // The optionals are explicitly default-constructed to allow the user to partially aggregrate-initialized
  // with only the options they care about
  struct Options {
    /**
     * @brief The polynomial order that should be used for the problem
     */
    int order = 1;
    /**
     * @brief The number of copies of the finite element collections (e.g. vector_dim = 2 or 3 for solid mechanics).
     * Defaults to scalar valued spaces.
     */
    int vector_dim = 1;
    /**
     * @brief The FECollection to use - defaults to an H1_FECollection
     */
    std::unique_ptr<mfem::FiniteElementCollection> coll = {};
    /**
     * @brief The DOF ordering that should be used interally by MFEM
     */
    mfem::Ordering::Type ordering = mfem::Ordering::byVDIM;
    /**
     * @brief The name of the field encapsulated by the state object
     */
    std::string name = "";

    /**
     * @brief Whether the underlying grid function should be allocated or not
     *
     * @note This should only be false for restart runs
     */
    bool alloc_local = true;
  };
  /**
   * Main constructor for building a new dual object
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] options The options specified, namely those relating to the order of the problem,
   * the dimension of the FESpace, the type of FEColl, the DOF ordering that should be used,
   * and the name of the field
   */
  FiniteElementDual(mfem::ParMesh& mesh, FiniteElementDual::Options&& options = {.order       = 1,
                                                                                 .vector_dim  = 1,
                                                                                 .coll        = {},
                                                                                 .ordering    = mfem::Ordering::byVDIM,
                                                                                 .name        = "",
                                                                                 .alloc_local = true})
      : state_(mesh, {.order      = options.order,
                      .vector_dim = options.vector_dim,
                      .coll       = std::move(options.coll),
                      .ordering   = options.ordering,
                      .name       = options.name,
                      .alloc_gf   = options.alloc_local}){};

  /**
   * @brief Minimal constructor for a FiniteElementDual given an already-existing grid function
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] gf The field for the dual to create (object does not take ownership)
   * @param[in] name The name of the field
   *
   * @note This is used mainly be the restart capability
   */
  FiniteElementDual(mfem::ParMesh& mesh, mfem::ParGridFunction& gf, const std::string& name = "")
      : state_(mesh, gf, name){};

  /**
   * @brief Minimal constructor for a FiniteElementDual given a finite element space
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] space The space to use for the finite element state. This space is deep copied into the new FE state
   * @param[in] name The name of the field
   */
  FiniteElementDual(mfem::ParMesh& mesh, mfem::ParFiniteElementSpace& space, const std::string& name = "")
      : state_(mesh, space, name){};

  /**
   * @brief Returns a non-owning reference to the internal mesh object
   */
  mfem::ParMesh& mesh() { return state_.mesh(); }

  /**
   * Returns a non-owning reference to the internal FESpace
   */
  mfem::ParFiniteElementSpace& space() { return state_.space(); }
  /// \overload
  const mfem::ParFiniteElementSpace& space() const { return state_.space(); }

  /**
   * @brief Returns a non-owning reference to the vector of true DOFs
   */
  mfem::HypreParVector& trueVec() { return state_.trueVec(); }

  /// \overload
  const mfem::HypreParVector& trueVec() const { return state_.trueVec(); }

  /**
   * @brief Returns a non-owning reference to the local degrees of freedom
   *
   * @return mfem::Vector& The local dof vector
   * @note While this is a grid function for plotting and parallelization, we only return a vector
   * type as the user should not use the interpolation capabilities of a grid function on the dual space
   */
  mfem::Vector& localVec() { return state_.gridFunc(); }

  /// @overload
  const mfem::Vector& localVec() const { return state_.gridFunc(); }

  /**
   * @brief Returns the name of the FEDual
   */
  std::string name() const { return state_.name(); }

  /**
   * @brief Initialize the true DOF vector by extracting true DOFs from the local
   * vector into the internal true DOF vector
   */
  void initializeTrueVec() { state_.initializeTrueVec(); }

  /**
   * @brief Set the local vector using the true DOF values
   */
  void distributeSharedDofs() { state_.distributeSharedDofs(); }

  /**
   * @brief Set the value of the dual to a scalar
   *
   * @param value The scalar to set
   * @return A reference to the modified dual
   *
   * @note This operates on the true dofs. In other words, if the value is different of different MPI ranks,
   * the rank which owns the DOF will set the value in the local vector.
   */
  FiniteElementDual& operator=(const double value)
  {
    state_ = value;
    distributeSharedDofs();
    return *this;
  }

  /**
   * Utility function for creating a tensor, e.g. mfem::HypreParVector,
   * mfem::ParBilinearForm, etc on the FESpace encapsulated by an FEDual object
   * @return An owning pointer to a heap-allocated tensor
   * @pre Tensor must have the constructor Tensor::Tensor(ParFiniteElementSpace*)
   */
  template <typename Tensor>
  std::unique_ptr<Tensor> createOnSpace()
  {
    return state_.createOnSpace<Tensor>();
  }

private:
  /**
   * @brief The underlying FE state
   */
  FiniteElementState state_;
};

}  // namespace serac

// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
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
#include <type_traits>
#include <variant>

#include "mfem.hpp"

#include "axom/sidre/core/MFEMSidreDataCollection.hpp"

namespace serac {

/**
 * @brief A sum type for encapsulating either a scalar or vector coeffient
 */
using GeneralCoefficient = std::variant<std::shared_ptr<mfem::Coefficient>, std::shared_ptr<mfem::VectorCoefficient>>;

/**
 * @brief convenience function for querying the type stored in a GeneralCoefficient
 */
inline bool is_scalar_valued(const GeneralCoefficient& coef)
{
  return std::holds_alternative<std::shared_ptr<mfem::Coefficient>>(coef);
}

/**
 * @brief convenience function for querying the type stored in a GeneralCoefficient
 */
inline bool is_vector_valued(const GeneralCoefficient& coef)
{
  return std::holds_alternative<std::shared_ptr<mfem::VectorCoefficient>>(coef);
}

/**
 * @brief Class for encapsulating the critical MFEM components of a solver
 *
 * Namely: Mesh, FiniteElementCollection, FiniteElementState,
 * GridFunction, and a Vector of the solution
 */
class FiniteElementState {
public:
  /**
   * @brief Structure for optionally configuring a FiniteElementState
   */
  // The optionals are explicitly default-constructed to allow the user to partially aggregrate-initialized
  // with only the options they care about
  struct Options {
    /**
     * @brief The polynomial order that should be used for the problem
     */
    int order = 1;
    /**
     * @brief The vector dimension for the FiniteElementSpace - defaults to the dimension of the mesh
     */
    std::optional<int> space_dim = {};
    /**
     * @brief The FECollection to use - defaults to an H1_FECollection
     */
    std::unique_ptr<mfem::FiniteElementCollection> coll = {};
    /**
     * The DOF ordering that should be used interally by MFEM
     */
    mfem::Ordering::Type ordering = mfem::Ordering::byVDIM;
    /**
     * @brief The name of the field encapsulated by the state object
     */
    std::string name = "";

    /**
     * @brief Whether to allocate memory for the GridFunction
     */
    bool allocate_gf = true;
  };

  /**
   * Main constructor for building a new state object
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] options The options specified, namely those relating to the order of the problem,
   * the dimension of the FESpace, the type of FEColl, the DOF ordering that should be used,
   * and the name of the field
   */
  FiniteElementState(mfem::ParMesh& mesh, Options&& options = {.order       = 1,
                                                               .space_dim   = {},
                                                               .coll        = {},
                                                               .ordering    = mfem::Ordering::byVDIM,
                                                               .name        = "",
                                                               .allocate_gf = true});

  /**
   * @brief Minimal constructor for a FiniteElementState given an already-existing field
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] gf The field for the state to create (object does not take ownership)
   * @param[in] name The name of the field
   */
  FiniteElementState(mfem::ParMesh& mesh, mfem::ParGridFunction& gf, const std::string& name = "");

  /**
   * Returns the MPI communicator for the state
   */
  MPI_Comm comm() const { return retrieve(space_).GetComm(); }

  /**
   * Returns a non-owning reference to the internal grid function
   */
  mfem::ParGridFunction&       gridFunc() { return retrieve(gf_); }
  const mfem::ParGridFunction& gridFunc() const { return retrieve(gf_); }

  /**
   * Returns a non-owning reference to the internal mesh object
   */
  mfem::ParMesh& mesh() { return retrieve(mesh_); }

  /**
   * Returns a non-owning reference to the internal FESpace
   */
  mfem::ParFiniteElementSpace& space() { return retrieve(space_); }

  /**
   * Returns a non-owning const reference to the internal FESpace
   */
  const mfem::ParFiniteElementSpace& space() const { return retrieve(space_); }

  /**
   * Returns a non-owning reference to the vector of true DOFs
   */
  mfem::HypreParVector& trueVec() { return true_vec_; }

  /**
   * Returns the name of the FEState (field)
   */
  std::string name() const { return name_; }

  /**
   * Projects a coefficient (vector or scalar) onto the field
   * @param[in] coef The coefficient to project
   */
  void project(const GeneralCoefficient& coef)
  {
    // The generic lambda parameter, auto&&, allows the component type (mfem::Coef or mfem::VecCoef)
    // to be deduced, and the appropriate version of ProjectCoefficient is dispatched.
    std::visit([this](auto&& concrete_coef) { retrieve(gf_).ProjectCoefficient(*concrete_coef); }, coef);
  }

  /**
   * Projects a coefficient (vector or scalar) onto the field
   * @param[in] coef The coefficient to project
   */
  void project(mfem::Coefficient& coef) { retrieve(gf_).ProjectCoefficient(coef); }

  /**
   * Projects a coefficient (vector or scalar) onto the field
   * @param[in] coef The coefficient to project
   */
  void project(mfem::VectorCoefficient& coef) { retrieve(gf_).ProjectCoefficient(coef); }

  /**
   * Initialize the true DOF vector by extracting true DOFs from the internal
   * grid function into the internal true DOF vector
   */
  void initializeTrueVec() { retrieve(gf_).GetTrueDofs(true_vec_); }

  /**
   * Set the internal grid function using the true DOF values
   */
  void distributeSharedDofs() { retrieve(gf_).SetFromTrueDofs(true_vec_); }

  /**
   * Utility function for creating a tensor, e.g. mfem::HypreParVector,
   * mfem::ParBilinearForm, etc on the FESpace encapsulated by an FEState object
   * @return An owning pointer to a heap-allocated tensor
   * @pre Tensor must have the constructor Tensor::Tensor(ParFiniteElementSpace*)
   */
  template <typename Tensor>
  std::unique_ptr<Tensor> createOnSpace()
  {
    static_assert(std::is_constructible_v<Tensor, mfem::ParFiniteElementSpace*>,
                  "Tensor must be constructible with a ptr to ParFESpace");
    return std::make_unique<Tensor>(&retrieve(space_));
  }

private:
  /**
   * @brief A helper type for uniform semantics over owning/non-owning pointers
   */
  template <typename T>
  using MaybeOwner = std::variant<T*, std::unique_ptr<T>>;

  /**
   * @brief Retrieves a reference to the underlying object in a MaybeOwner
   * @param[in] obj The object to dereference
   */
  template <typename T>
  static T& retrieve(MaybeOwner<T>& obj)
  {
    return std::visit([](auto&& ptr) -> T& { return *ptr; }, obj);
  }
  /// @overload
  template <typename T>
  static const T& retrieve(const MaybeOwner<T>& obj)
  {
    return std::visit([](auto&& ptr) -> const T& { return *ptr; }, obj);
  }

  // Allows for copy/move assignment
  MaybeOwner<mfem::ParMesh> mesh_;
  // Must be const as FESpaces store a const reference to their FEColls
  MaybeOwner<const mfem::FiniteElementCollection> coll_;
  MaybeOwner<mfem::ParFiniteElementSpace>         space_;
  MaybeOwner<mfem::ParGridFunction>               gf_;
  mfem::HypreParVector                            true_vec_;
  std::string                                     name_ = "";
};

/**
 * @brief Manages the lifetimes of FEState objects such that restarts are abstracted
 * from physics modules
 */
class StateManager {
public:
  /**
   * @brief Initializes the StateManager with a sidre DataStore (into which state will be written/read)
   * @param[in] ds The DataStore to use
   * @param[in] cycle_to_load The cycle to load - required for restarts
   */
  static void initialize(axom::sidre::DataStore& ds, const std::optional<int> cycle_to_load = {});

  /**
   * @brief Factory method for creating a new FEState object, signature is identical to FEState constructor
   * @param[in] mesh The problem mesh
   * @param[in] options Configuration options for the FEState, if a new state is created
   * @see FiniteElementState::FiniteElementState
   * @note If this is a restart then the options (except for the name) will be ignored
   */
  static FiniteElementState newState(mfem::ParMesh& mesh, FiniteElementState::Options&& options = {});

  /**
   * @brief Updates the Conduit Blueprint state in the datastore and saves to a file
   * @param[in] t The current sim time
   * @param[in] cycle The current iteration number of the simulation
   */
  static void step(const double t, const int cycle);

  /**
   * @brief Resets the underlying global datacollection object
   */
  static void reset() { datacoll_.reset(); };

private:
  /**
   * @brief The datacollection instance
   */
  static std::optional<axom::sidre::MFEMSidreDataCollection> datacoll_;
  /**
   * @brief Whether this simulation has been restarted from another simulation
   */
  static bool is_restart_;
};

}  // namespace serac

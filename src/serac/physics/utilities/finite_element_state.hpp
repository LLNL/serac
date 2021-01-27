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
#include <type_traits>
#include <variant>

#include "mfem.hpp"

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
     * The polynomial order that should be used for the problem
     */
    int order = 1;
    /**
     * The vector dimension for the FiniteElementSpace - defaults to the dimension of the mesh
     */
    std::optional<int> space_dim = {};
    /**
     * The FECollection to use - defaults to an H1_FECollection
     */
    std::unique_ptr<mfem::FiniteElementCollection> coll = {};
    /**
     * The DOF ordering that should be used interally by MFEM
     */
    mfem::Ordering::Type ordering = mfem::Ordering::byVDIM;
    /**
     * The name of the field encapsulated by the state object
     */
    std::string name = "";
  };

  /**
   * Main constructor for building a new state object
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] options The options specified, namely those relating to the order of the problem,
   * the dimension of the FESpace, the type of FEColl, the DOF ordering that should be used,
   * and the name of the field
   */
  FiniteElementState(mfem::ParMesh& mesh,
                     Options&&      options = {
                         .order = 1, .space_dim = {}, .coll = {}, .ordering = mfem::Ordering::byVDIM, .name = ""});

  /**
   * Returns the MPI communicator for the state
   */
  MPI_Comm comm() const { return space_.GetComm(); }

  /**
   * Returns a non-owning reference to the internal grid function
   */
  mfem::ParGridFunction& gridFunc() { return *gf_; }
  /// \overload
  const mfem::ParGridFunction& gridFunc() const { return *gf_; }

  /**
   * Returns a non-owning reference to the internal mesh object
   */
  mfem::ParMesh& mesh() { return mesh_; }

  /**
   * Returns a non-owning reference to the internal FESpace
   */
  mfem::ParFiniteElementSpace& space() { return space_; }
  /// \overload
  const mfem::ParFiniteElementSpace& space() const { return space_; }

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
    std::visit([this](auto&& concrete_coef) { gf_->ProjectCoefficient(*concrete_coef); }, coef);
  }
  /// \overload
  void project(mfem::Coefficient& coef) { gf_->ProjectCoefficient(coef); }
  /// \overload
  void project(mfem::VectorCoefficient& coef) { gf_->ProjectCoefficient(coef); }

  /**
   * Initialize the true DOF vector by extracting true DOFs from the internal
   * grid function into the internal true DOF vector
   */
  void initializeTrueVec() { gf_->GetTrueDofs(true_vec_); }

  /**
   * Set the internal grid function using the true DOF values
   */
  void distributeSharedDofs() { gf_->SetFromTrueDofs(true_vec_); }

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
    return std::make_unique<Tensor>(&space_);
  }

private:
  // Allows for copy/move assignment
  std::reference_wrapper<mfem::ParMesh>          mesh_;
  std::unique_ptr<mfem::FiniteElementCollection> coll_;
  mfem::ParFiniteElementSpace                    space_;
  std::unique_ptr<mfem::ParGridFunction>         gf_;
  mfem::HypreParVector                           true_vec_;
  std::string                                    name_ = "";
};

}  // namespace serac

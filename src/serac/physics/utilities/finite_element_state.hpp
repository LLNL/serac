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

#include "serac/physics/utilities/variant.hpp"

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
     * @brief The number of copies of the finite element collections (e.g. vector_dim = 2 or 3 for solid mechanics).
     * Defaults to scalar valued spaces.
     */
    int vector_dim = 1;
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
     * @brief Whether the GridFunction should be allocated (and owned by the FEState object)
     */
    bool alloc_gf = true;
  };

  /**
   * Main constructor for building a new state object
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] options The options specified, namely those relating to the order of the problem,
   * the dimension of the FESpace, the type of FEColl, the DOF ordering that should be used,
   * and the name of the field
   */
  FiniteElementState(
      mfem::ParMesh& mesh,
      Options&&      options = {
          .order = 1, .vector_dim = 1, .coll = {}, .ordering = mfem::Ordering::byVDIM, .name = "", .alloc_gf = true});

  /**
   * @brief Minimal constructor for a FiniteElementState given an already-existing field
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] gf The field for the state to create (object does not take ownership)
   * @param[in] name The name of the field
   */
  FiniteElementState(mfem::ParMesh& mesh, mfem::ParGridFunction& gf, const std::string& name = "");

  /**
   * @brief Minimal constructor for a FiniteElementState given an already-existing state
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] fe_state The state for the new state to copy
   * @param[in] name The name of the field
   */
  FiniteElementState(mfem::ParMesh& mesh, FiniteElementState& fe_state, const std::string& name = "");

  /**
   * Returns the MPI communicator for the state
   */
  MPI_Comm comm() const { return detail::retrieve(space_).GetComm(); }

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
   * Returns a non-owning reference to the internal mesh object
   */
  mfem::ParMesh& mesh() { return mesh_; }

  /**
   * Returns a non-owning reference to the internal FESpace
   */
  mfem::ParFiniteElementSpace& space() { return detail::retrieve(space_); }
  /// \overload
  const mfem::ParFiniteElementSpace& space() const { return detail::retrieve(space_); }

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
   * Initialize the true DOF vector by extracting true DOFs from the internal
   * grid function into the internal true DOF vector
   */
  void initializeTrueVec() { detail::retrieve(gf_).GetTrueDofs(true_vec_); }

  /**
   * Set the internal grid function using the true DOF values
   */
  void distributeSharedDofs() { detail::retrieve(gf_).SetFromTrueDofs(true_vec_); }

  /**
   * @brief Set a finite element state to a constant value
   *
   * @param value The constant to set the finite element state to
   * @return The modified finite element state
   */
  FiniteElementState& operator=(const double value);

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
    return std::make_unique<Tensor>(&detail::retrieve(space_));
  }

private:
  /**
   * @brief A reference to the mesh object on which the field is defined
   */
  std::reference_wrapper<mfem::ParMesh> mesh_;
  /**
   * @brief Possibly-owning handle to the FiniteElementCollection, as it is owned
   * by the FiniteElementState in a normal run and by the MFEMSidreDataCollection
   * in a restart run
   * @note Must be const as FESpaces store a const reference to their FEColls
   */
  detail::MaybeOwningPointer<const mfem::FiniteElementCollection> coll_;
  /**
   * @brief Possibly-owning handle to the FiniteElementCollection, as it is owned
   * by the FiniteElementState in a normal run and by the MFEMSidreDataCollection
   * in a restart run
   */
  detail::MaybeOwningPointer<mfem::ParFiniteElementSpace> space_;
  /**
   * @brief Possibly-owning handle to the ParGridFunction, as it is owned
   * by the FiniteElementState in a normal run and by the MFEMSidreDataCollection
   * in a restart run
   */
  detail::MaybeOwningPointer<mfem::ParGridFunction> gf_;
  mfem::HypreParVector                              true_vec_;
  std::string                                       name_ = "";
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

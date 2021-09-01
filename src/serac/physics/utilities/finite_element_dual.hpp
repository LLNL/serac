// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file finite_element_dual.hpp
 *
 * @brief This contains a class that represents the space of topological duals sampled
 * on a discrete finite element basis, i.e. the space of residuals and sensitivities.
 */

#pragma once

#include "mfem.hpp"

namespace serac {

/**
 * @brief Class for encapsulating the topological dual space of a finite element space (i.e. the
 * space of linear forms) as applied to a specific basis set
 *
 * @note While a grid function is provided by this class, interpolation operations are not
 * defined in this representation of the dual space and should be used with caution
 *
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
     * The DOF ordering that should be used interally by MFEM
     */
    mfem::Ordering::Type ordering = mfem::Ordering::byVDIM;
    /**
     * @brief The name of the field encapsulated by the state object
     */
    std::string name = "";
  };
  /**
   * Main constructor for building a new dual object
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] options The options specified, namely those relating to the order of the problem,
   * the dimension of the FESpace, the type of FEColl, the DOF ordering that should be used,
   * and the name of the field
   */
  FiniteElementDual(mfem::ParMesh& mesh,
                    Options&&      options = {
                        .order = 1, .vector_dim = 1, .coll = {}, .ordering = mfem::Ordering::byVDIM, .name = ""});

  /**
   * @brief Minimal constructor for a FiniteElementDual given a finite element space
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] space The space to use for the finite element state. This space is deep copied into the new FE state
   * @param[in] name The name of the field
   */
  FiniteElementDual(mfem::ParMesh& mesh, mfem::ParFiniteElementSpace& space, const std::string& name = "");

  /**
   * Returns a non-owning reference to the internal mesh object
   */
  mfem::ParMesh& mesh() { return mesh_; }

  /**
   * Returns a non-owning reference to the internal FESpace
   */
  mfem::ParFiniteElementSpace& space() { return *space_; }
  /// \overload
  const mfem::ParFiniteElementSpace& space() const { return *space_; }

  /**
   * Returns a non-owning reference to the vector of true DOFs
   */
  mfem::HypreParVector& trueVec() { return true_vec_; }

  /// \overload
  const mfem::HypreParVector& trueVec() const { return true_vec_; }

  /**
   * @brief Returns a non-owning reference to the local degrees of freedom
   *
   * @return mfem::Vector& The local dof vector
   */
  mfem::Vector& localVec() { return local_vec_; }

  /// @overload
  const mfem::Vector& localVec() const { return local_vec_; }

  /**
   * Returns the name of the FEDual
   */
  std::string name() const { return name_; }

  /**
   * Initialize the true DOF vector by extracting true DOFs from the local
   * vector into the internal true DOF vector
   */
  void initializeTrueVec();

  /**
   * Set the local vector using the true DOF values
   */
  void distributeSharedDofs();

  /**
   * Utility function for creating a tensor, e.g. mfem::HypreParVector,
   * mfem::ParBilinearForm, etc on the FESpace encapsulated by an FEDual object
   * @return An owning pointer to a heap-allocated tensor
   * @pre Tensor must have the constructor Tensor::Tensor(ParFiniteElementSpace*)
   */
  template <typename Tensor>
  std::unique_ptr<Tensor> createOnSpace()
  {
    static_assert(std::is_constructible_v<Tensor, mfem::ParFiniteElementSpace*>,
                  "Tensor must be constructible with a ptr to ParFESpace");
    return std::make_unique<Tensor>(space_.get());
  }

  /**
   * @brief Set a finite element dual to a constant value
   *
   * @param value The constant to set the finite element dual to
   * @return The modified finite element dual
   * @note This sets the true degrees of freedom and then broadcasts to the shared grid function entries. This means
   * that if a different value is given on different processors, a shared DOF will be set to the owning processor value.
   */
  FiniteElementDual& operator=(const double value);

private:
  /**
   * @brief A reference to the mesh object on which the field is defined
   */
  std::reference_wrapper<mfem::ParMesh> mesh_;
  /**
   * @brief Pointer to the FiniteElementCollection
   */
  std::unique_ptr<mfem::FiniteElementCollection> coll_;
  /**
   * @brief Pointer to the FiniteElementSpace
   */
  std::unique_ptr<mfem::ParFiniteElementSpace> space_;
  /**
   * @brief Local (or L) vector. This includes the shared dofs
   * on each MPI rank.
   */
  mfem::Vector local_vec_;
  /**
   * @brief The true degree of freedom vector. Each entry is only listed
   * on one MPI rank.
   */
  mfem::HypreParVector true_vec_;
  /**
   * @brief Name of the finite element dual.
   */
  std::string name_ = "";
};

}  // namespace serac

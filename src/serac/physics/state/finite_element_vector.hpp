// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file finite_element_vector.hpp
 *
 * @brief This file contains the declaration of structure that manages vectors
 * derived from an MFEM finite element space
 */

#pragma once

#include "mfem.hpp"

#include "serac/infrastructure/variant.hpp"

namespace serac {

/**
 * @brief Class for encapsulating the data associated with a vector derived
 * from a MFEM finite element space. Specifically, it contains the information
 * needed for both primal finite element state fields and dual finite element vectors.
 *
 * Namely: Mesh, FiniteElementCollection, FiniteElementVector,
 * GridFunction, and a distributed vector of the solution
 */
class FiniteElementVector {
public:
  /**
   * @brief Structure for optionally configuring a FiniteElementVector
   * @note The options are explicitly default-constructed to allow the user to partially aggregrate-initialized
   * with only the options they care about
   */
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
     * @brief A bool denoting if the grid function is managed by sidre
     * @note This should only be true if calling a constructor from the StateManager class
     */
    bool managed_by_sidre = false;
  };

  /**
   * @brief Main constructor for building a new finite element vector
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] options The options specified, namely those relating to the order of the problem,
   * the dimension of the FESpace, the type of FEColl, the DOF ordering that should be used,
   * and the name of the field
   */
  FiniteElementVector(mfem::ParMesh& mesh, Options&& options = {.order            = 1,
                                                                .vector_dim       = 1,
                                                                .coll             = {},
                                                                .ordering         = mfem::Ordering::byVDIM,
                                                                .name             = "",
                                                                .managed_by_sidre = false});

  /**
   * @brief Minimal constructor for a FiniteElementVector given an already-existing field
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] gf The field for the state to create (object does not take ownership)
   * @param[in] name The name of the field
   */
  FiniteElementVector(mfem::ParMesh& mesh, mfem::ParGridFunction& gf, const std::string& name = "");

  /**
   * @brief Minimal constructor for a FiniteElementVector given a finite element space
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] space The space to use for the finite element state. This space is deep copied into the new FE state
   * @param[in] name The name of the field
   */
  FiniteElementVector(mfem::ParMesh& mesh, mfem::ParFiniteElementSpace& space, const std::string& name = "");

  /**
   * @brief Delete the default copy constructor
   */
  FiniteElementVector(const FiniteElementVector&) = delete;

  /**
   * @brief Move construct a new Finite Element Vector object
   *
   * @param[in] input_vector The input vector used for construction
   */
  FiniteElementVector(FiniteElementVector&& input_vector);

  /**
   * @brief Returns the MPI communicator for the state
   * @return The underlying MPI communicator
   */
  MPI_Comm comm() const { return detail::retrieve(space_).GetComm(); }

  /**
   * @brief Returns a non-owning reference to the internal mesh object
   * @return The underlying mesh
   */
  mfem::ParMesh& mesh() { return mesh_; }

  /**
   * @brief Returns a non-owning reference to the internal FESpace
   * @return The underlying finite element space
   */
  mfem::ParFiniteElementSpace& space() { return detail::retrieve(space_); }
  /// \overload
  const mfem::ParFiniteElementSpace& space() const { return detail::retrieve(space_); }

  /**
   * @brief Returns a non-owning reference to the vector of true DOFs
   * @return The underlying true degree of freedom vector
   * @note This is a "true dof" vector in the standard MFEM sense. Each degree of freedom is on fully independent
   * (e.g. not constrained by non-conforming meshes) and exists on exactly one MPI rank. Please see the
   * <a href="https://mfem.org/pri-dual-vec/">MFEM</a> and
   * <a href="https://libceed.readthedocs.io/en/latest/libCEEDapi/#terminology-and-notation">CEED</a> documentation for
   * more details.
   */
  mfem::HypreParVector& trueVec() { return true_vec_; }
  /// \overload
  const mfem::HypreParVector& trueVec() const { return true_vec_; }

  /**
   * @brief Returns the name of the FEState (field)
   * @return The name of the finite element vector
   */
  std::string name() const { return name_; }

  /**
   * @brief Set a finite element state to a constant value
   *
   * @param value The constant to set the finite element state to
   * @return The modified finite element state
   * @note This sets the true degrees of freedom and then broadcasts to the shared grid function entries. This means
   * that if a different value is given on different processors, a shared DOF will be set to the owning processor value.
   */
  FiniteElementVector& operator=(const double value);

  /**
   * @brief Distribute dofs the internal grid function (local dofs) using the true DOF values
   */
  virtual void distributeSharedDofs() = 0;

  /**
   * @brief Initialize the true DOF vector using the internal grid function
   */
  virtual void initializeTrueVec() = 0;

  /**
   * @brief Utility function for creating a tensor, e.g. mfem::HypreParVector,
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

  /**
   * @brief Destroy the Finite Element Vector object
   */
  virtual ~FiniteElementVector() {}

protected:
  /**
   * @brief A reference to the mesh object on which the field is defined
   */
  std::reference_wrapper<mfem::ParMesh> mesh_;
  /**
   * @brief Possibly-owning handle to the FiniteElementCollection, as it is owned
   * by the FiniteElementVector in a normal run and by the MFEMSidreDataCollection
   * in a restart run
   * @note Must be const as FESpaces store a const reference to their FEColls
   */
  detail::MaybeOwningPointer<const mfem::FiniteElementCollection> coll_;
  /**
   * @brief Possibly-owning handle to the mfem::ParFiniteElementSpace, as it is owned
   * by the FiniteElementVector in a normal run and by the MFEMSidreDataCollection
   * in a restart run
   */
  detail::MaybeOwningPointer<mfem::ParFiniteElementSpace> space_;
  /**
   * @brief Possibly-owning handle to the ParGridFunction, as it is owned
   * by the FiniteElementVector in a normal run and by the MFEMSidreDataCollection
   * in a restart run
   */
  detail::MaybeOwningPointer<mfem::ParGridFunction> gf_;
  /**
   * @brief The hypre vector containing the true degrees of freedom
   * @note Each entry in this vector is owned by exactly one MPI rank
   */
  mfem::HypreParVector true_vec_;
  /**
   * @brief The name of the finite element vector
   */
  std::string name_ = "";
};

/**
 * @brief Find the average value of a finite element vector across all dofs
 *
 * @param fe_vector The state variable to compute the average of
 * @return The average value
 * @note This acts on the actual scalar degree of freedom values, not the interpolated shape function values. This
 * implies these may or may not be nodal averages depending on the choice of finite element basis.
 */
double avg(const FiniteElementVector& fe_vector);

/**
 * @brief Find the max value of a finite element vector across all dofs
 *
 * @param fe_vector The state variable to compute a max of
 * @return The max value
 * @note This acts on the actual scalar degree of freedom values, not the interpolated shape function values. This
 * implies these may or may not be nodal averages depending on the choice of finite element basis.
 */
double max(const FiniteElementVector& fe_vector);

/**
 * @brief Find the min value of a finite element vector across all dofs
 *
 * @param fe_vector The state variable to compute a min of
 * @return The min value
 * @note This acts on the actual scalar degree of freedom values, not the interpolated shape function values. This
 * implies these may or may not be nodal averages depending on the choice of finite element basis.
 */
double min(const FiniteElementVector& fe_vector);

}  // namespace serac

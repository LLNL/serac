// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
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

#include <optional>

#include "mfem.hpp"

#include "serac/infrastructure/variant.hpp"

namespace serac {

/**
 * @brief A sum type for encapsulating either a scalar or vector coeffient
 */
using GeneralCoefficient = variant<std::shared_ptr<mfem::Coefficient>, std::shared_ptr<mfem::VectorCoefficient>>;

/**
 * @brief Class for encapsulating the data associated with a vector derived
 * from a MFEM finite element space. Specifically, it contains the information
 * needed for both primal finite element state fields and dual finite element vectors.
 *
 * Namely: Mesh, FiniteElementCollection, FiniteElementSpace, name, and a HypreParVector
 * containing the true degrees of freedom for the field.
 */
class FiniteElementVector : public mfem::HypreParVector {
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
  };

  /**
   * @brief Main constructor for building a new finite element vector
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] options The options specified, namely those relating to the order of the problem,
   * the dimension of the FESpace, the type of FEColl, the DOF ordering that should be used,
   * and the name of the field
   */
  FiniteElementVector(mfem::ParMesh& mesh,
                      Options&&      options = {
                          .order = 1, .vector_dim = 1, .coll = {}, .ordering = mfem::Ordering::byVDIM, .name = ""});

  /**
   * @brief Minimal constructor for a FiniteElementVector given a finite element space
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] space The space to use for the finite element state. This space is deep copied into the new FE state
   * @param[in] name The name of the field
   */
  FiniteElementVector(mfem::ParMesh& mesh, const mfem::ParFiniteElementSpace& space, const std::string& name = "");

  /**
   * @brief Copy constructor
   *
   * @param[in] rhs The input vector used for construction
   */
  FiniteElementVector(const FiniteElementVector& rhs) : FiniteElementVector(rhs.mesh_.get(), *rhs.space_, rhs.name_) {}

  /**
   * @brief Move construct a new Finite Element Vector object
   *
   * @param[in] rhs The input vector used for construction
   */
  FiniteElementVector(FiniteElementVector&& rhs);

  /**
   * @brief Copy assignment
   *
   * @param rhs The right hand side input vector
   * @return The assigned FiniteElementVector
   */
  FiniteElementVector& operator=(const FiniteElementVector& rhs);

  /**
   * @brief Move assignment
   *
   * @param rhs The right hand side input vector
   * @return The move assigned input vector
   */
  FiniteElementVector& operator=(FiniteElementVector&& rhs);

  /**
   * @brief Copy assignment from a hypre par vector
   *
   * @param rhs The rhs input hypre par vector
   * @return The copy assigned input vector
   */
  FiniteElementVector& operator=(const mfem::HypreParVector& rhs);

  /**
   * @brief Returns the MPI communicator for the state
   * @return The underlying MPI communicator
   */
  MPI_Comm comm() const { return space_->GetComm(); }

  /**
   * @brief Returns a non-owning reference to the internal mesh object
   * @return The underlying mesh
   */
  mfem::ParMesh& mesh() { return mesh_; }
  /// \overload
  const mfem::ParMesh& mesh() const { return mesh_; }

  /**
   * @brief Returns a non-owning reference to the internal FESpace
   * @return The underlying finite element space
   */
  mfem::ParFiniteElementSpace& space() { return *space_; }
  /// \overload
  const mfem::ParFiniteElementSpace& space() const { return *space_; }

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
   * @brief Destroy the Finite Element Vector object
   */
  virtual ~FiniteElementVector() {}

protected:
  /**
   * @brief A reference to the mesh object on which the field is defined
   */
  std::reference_wrapper<mfem::ParMesh> mesh_;

  /**
   * @brief Handle to the FiniteElementCollection, which is owned by MFEMSidreDataCollection
   * @note Must be const as FESpaces store a const reference to their FEColls
   */
  std::unique_ptr<mfem::FiniteElementCollection> coll_;

  /**
   * @brief Handle to the mfem::ParFiniteElementSpace, which is owned by MFEMSidreDataCollection
   */
  std::unique_ptr<mfem::ParFiniteElementSpace> space_;

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

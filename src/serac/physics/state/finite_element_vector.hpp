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
#include "axom/fmt.hpp"

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
   */
  FiniteElementVector(const FiniteElementVector& rhs);

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
  MPI_Comm comm() const { return space_->GetComm(); }

  /**
   * @brief Returns a non-owning reference to the internal mesh object
   * @return The underlying mesh
   */
  mfem::ParMesh& mesh() { return mesh_; }

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

  FiniteElementVector& operator=(const mfem::HypreParVector& input)
  {
    true_vec_ = input;
    return *this;
  }

  FiniteElementVector& operator+=(const mfem::Vector& input)
  {
    true_vec_ += input;
    return *this;
  }

  FiniteElementVector& operator-=(const mfem::Vector& input)
  {
    true_vec_ -= input;
    return *this;
  }

  FiniteElementVector& operator*=(const double input)
  {
    true_vec_ *= input;
    return *this;
  }

  double& operator()(int index) { return true_vec_(index); }

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
    return std::make_unique<Tensor>(space_.get());
  }

  void project(mfem::VectorCoefficient& coef, mfem::Array<int>& dof_list)
  {
    mfem::ParGridFunction grid_function = gridFunction();
    grid_function.ProjectCoefficient(coef, dof_list);
    initializeTrueVec(grid_function);
  }

  void project(mfem::Coefficient& coef, mfem::Array<int>& dof_list, std::optional<int> component = {})
  {
    mfem::ParGridFunction grid_function = gridFunction();
    axom::fmt::print("grid function size: {}\n", grid_function.Size());
    if (component) {
      grid_function.ProjectCoefficient(coef, dof_list, *component);
    } else {
      grid_function.ProjectCoefficient(coef, dof_list, *component);
    }
    initializeTrueVec(grid_function);
  }

  /**
   * Projects a coefficient (vector or scalar) onto the field
   * @param[in] coef The coefficient to project
   */
  void project(const GeneralCoefficient& coef)
  {
    mfem::ParGridFunction grid_function = gridFunction();

    // The generic lambda parameter, auto&&, allows the component type (mfem::Coef or mfem::VecCoef)
    // to be deduced, and the appropriate version of ProjectCoefficient is dispatched.
    visit(
        [this, &grid_function](auto&& concrete_coef) {
          grid_function.ProjectCoefficient(*concrete_coef);
          initializeTrueVec(grid_function);
        },
        coef);
  }
  /// \overload
  void project(mfem::Coefficient& coef)
  {
    mfem::ParGridFunction grid_function = gridFunction();
    grid_function.ProjectCoefficient(coef);
    initializeTrueVec(grid_function);
  }
  /// \overload
  void project(mfem::VectorCoefficient& coef)
  {
    mfem::ParGridFunction grid_function = gridFunction();
    grid_function.ProjectCoefficient(coef);
    initializeTrueVec(grid_function);
  }

  void projectBdr(mfem::Coefficient& coef, const mfem::Array<int>& markers)
  {
    mfem::ParGridFunction grid_function = gridFunction();
    // markers should be const param in mfem, but it's not
    grid_function.ProjectBdrCoefficient(coef, const_cast<mfem::Array<int>&>(markers));
    initializeTrueVec(grid_function);
  }

  void projectBdr(mfem::VectorCoefficient& coef, const mfem::Array<int>& markers)
  {
    mfem::ParGridFunction grid_function = gridFunction();
    // markers should be const param in mfem, but it's not
    grid_function.ProjectBdrCoefficient(coef, const_cast<mfem::Array<int>&>(markers));
    initializeTrueVec(grid_function);
  }

  void initialize(const mfem::ParGridFunction& grid_function) { initializeTrueVec(grid_function); }

  mfem::ParGridFunction gridFunction() const
  {
    axom::fmt::print("space size: {}\n", space_->GetVSize());
    mfem::ParGridFunction grid_function(space_.get());
    axom::fmt::print("grid function printed: {}\n", grid_function.Size());
    distributeSharedDofs(grid_function);
    axom::fmt::print("grid function printed 2: {}\n", grid_function.Size());
    return grid_function;
  }

  void gridFunction(mfem::ParGridFunction& grid_function) const { distributeSharedDofs(grid_function); }

  int vectorDim()
  {
    mfem::ParGridFunction grid_function = gridFunction();
    return grid_function.VectorDim();
  }

  /**
   * @brief Find the average value of a finite element vector across all dofs
   *
   * @param fe_vector The state variable to compute the average of
   * @return The average value
   * @note This acts on the actual scalar degree of freedom values, not the interpolated shape function values. This
   * implies these may or may not be nodal averages depending on the choice of finite element basis.
   */
  friend double avg(const FiniteElementVector& fe_vector);

  /**
   * @brief Find the max value of a finite element vector across all dofs
   *
   * @param fe_vector The state variable to compute a max of
   * @return The max value
   * @note This acts on the actual scalar degree of freedom values, not the interpolated shape function values. This
   * implies these may or may not be nodal averages depending on the choice of finite element basis.
   */
  friend double max(const FiniteElementVector& fe_vector);

  /**
   * @brief Find the min value of a finite element vector across all dofs
   *
   * @param fe_vector The state variable to compute a min of
   * @return The min value
   * @note This acts on the actual scalar degree of freedom values, not the interpolated shape function values. This
   * implies these may or may not be nodal averages depending on the choice of finite element basis.
   */
  friend double min(const FiniteElementVector& fe_vector);

  operator mfem::HypreParVector &() { return true_vec_; }

  const mfem::HypreParVector& vector() const { return true_vec_; }

  mfem::HypreParVector& vector() { return true_vec_; }

  /**
   * @brief Destroy the Finite Element Vector object
   */
  virtual ~FiniteElementVector() {}

protected:
  /**
   * @brief Distribute dofs the internal grid function (local dofs) using the true DOF values
   */
  virtual void distributeSharedDofs(mfem::ParGridFunction& grid_function) const = 0;

  /**
   * @brief Initialize the true DOF vector using the internal grid function
   */
  virtual void initializeTrueVec(const mfem::ParGridFunction& grid_function) = 0;

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

  mfem::HypreParVector true_vec_;
};

double norm(const FiniteElementVector& state, const double p = 2);

}  // namespace serac

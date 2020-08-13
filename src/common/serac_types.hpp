// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file serac_types.hpp
 *
 * @brief This file contains common serac data structures
 */

#ifndef SERAC_TYPES
#define SERAC_TYPES

#include <memory>
#include <optional>
#include <type_traits>
#include <variant>

#include "common/logger.hpp"
#include "mfem.hpp"

namespace serac {
/**
 * @brief Output file type associated with a solver
 */
enum class OutputType
{
  GLVis,
  VisIt
};

/**
 * @brief Timestep method of a solver
 */
enum class TimestepMethod
{
  BackwardEuler,
  SDIRK33,
  ForwardEuler,
  RK2,
  RK3SSP,
  RK4,
  GeneralizedAlpha,
  ImplicitMidpoint,
  SDIRK23,
  SDIRK34,
  QuasiStatic
};

/**
 * @brief Linear solution method
 */
enum class LinearSolver
{
  CG,
  GMRES,
  MINRES
};

/**
 * @brief Preconditioning method
 */
enum class Preconditioner
{
  Jacobi,
  BoomerAMG
};

/**
 * @brief Abstract multiphysics coupling scheme
 */
enum class CouplingScheme
{
  OperatorSplit,
  FixedPoint,
  FullyCoupled
};

/**
 * @brief Parameters for a linear solution scheme
 */
struct LinearSolverParameters {
  /**
   * @brief Relative tolerance
   */
  double rel_tol;

  /**
   * @brief Absolute tolerance
   */
  double abs_tol;

  /**
   * @brief Debugging print level
   */
  int print_level;

  /**
   * @brief Maximum number of iterations
   */
  int max_iter;

  /**
   * @brief Linear solver selection
   */
  LinearSolver lin_solver;

  /**
   * @brief Preconditioner selection
   */
  Preconditioner prec;
};

/**
 * @brief Nonlinear solution scheme parameters
 */
struct NonlinearSolverParameters {
  /**
   * @brief Relative tolerance
   */
  double rel_tol;

  /**
   * @brief Absolute tolerance
   */
  double abs_tol;

  /**
   * @brief Maximum number of iterations
   */
  int max_iter;

  /**
   * @brief Debug print level
   */
  int print_level;
};

// Git will git confused if the order of the classes are switched, so leave this in until merge
using BCCoef = std::variant<std::shared_ptr<mfem::Coefficient>, std::shared_ptr<mfem::VectorCoefficient>>;

/**
 * Structure for optionally configuring a FiniteElementState
 */
// The optionals are explicitly default-constructed to allow the user to partially aggregrate-initialized
// with only the options they care about
struct FESOptions {
  /**
   * The vector dimension for the FiniteElementSpace - defaults to the dimension of the mesh
   */
  std::optional<int> space_dim = std::optional<int>();
  /**
   * The FECollection to use - defaults to an H1_FECollection
   */
  std::optional<std::unique_ptr<mfem::FiniteElementCollection>> coll =
      std::optional<std::unique_ptr<mfem::FiniteElementCollection>>();
  /**
   * The DOF ordering that should be used interally by MFEM
   */
  mfem::Ordering::Type ordering = mfem::Ordering::byNODES;
  /**
   * The name of the field encapsulated by the state object
   */
  std::string name = "";
};

/**
 * Class for encapsulating the critical MFEM components of a solver
 * namely Mesh, FiniteElementCollection, FiniteElementState,
 * GridFunction, and a Vector of the solution
 */
class FiniteElementState {
public:
  /**
   * Main constructor for building a new state object
   * @param[in] order The order of the problem
   * @param[in] pmesh The problem mesh
   * @param[in] options The options specified, namely those relating to the dimension of the FESpace, the type of
   * FEColl, the DOF ordering that should be used, and the name of the field
   */
  FiniteElementState(const int order, std::shared_ptr<mfem::ParMesh> pmesh, FESOptions&& options = FESOptions());

  /**
   * Returns the MPI communicator for the state
   */
  MPI_Comm comm() const { return space_.GetComm(); }

  /**
   * Returns a non-owning reference to the internal grid function
   */
  mfem::ParGridFunction& gridFunc() { return *gf_; }

  /**
   * Returns a non-owning reference to the internal mesh object
   */
  mfem::ParMesh& mesh() { return *mesh_; }

  /**
   * Returns a non-owning reference to the internal FESpace
   */
  mfem::ParFiniteElementSpace& space() { return space_; }

  /**
   * Returns a non-owning const reference to the internal FESpace
   */
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
  void project(const BCCoef& coef)
  {
    // The generic lambda parameter, auto&&, allows the component type (mfem::Coef or mfem::VecCoef)
    // to be deduced, and the appropriate version of ProjectCoefficient is dispatched.
    std::visit([this](auto&& concrete_coef) { gf_->ProjectCoefficient(*concrete_coef); }, coef);
  }

  /**
   * Projects a coefficient (vector or scalar) onto the field
   * @param[in] coef The coefficient to project
   */
  void project(mfem::Coefficient& coef) { gf_->ProjectCoefficient(coef); }

  /**
   * Projects a coefficient (vector or scalar) onto the field
   * @param[in] coef The coefficient to project
   */
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
  std::unique_ptr<Tensor> createTensorOnSpace()
  {
    static_assert(std::is_constructible_v<Tensor, mfem::ParFiniteElementSpace*>,
                  "Tensor must be constructible with a ptr to ParFESpace");
    return std::make_unique<Tensor>(&space_);
  }

private:
  std::shared_ptr<mfem::ParMesh>                 mesh_;
  std::unique_ptr<mfem::FiniteElementCollection> coll_;
  mfem::ParFiniteElementSpace                    space_;
  std::unique_ptr<mfem::ParGridFunction>         gf_;
  mfem::HypreParVector                           true_vec_;
  std::string                                    name_ = "";
};

}  // namespace serac

#endif

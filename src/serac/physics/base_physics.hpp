// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file base_physics.hpp
 *
 * @brief The base interface class for a generic PDE solver
 */

#pragma once

#include <functional>
#include <memory>

#include "mfem.hpp"
#include "axom/sidre.hpp"

#include "serac/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "serac/numerics/equation_solver.hpp"
#include "serac/physics/state/finite_element_state.hpp"
#include "serac/physics/state/finite_element_dual.hpp"
#include "serac/physics/common.hpp"

namespace serac {

/**
 * @brief This is the abstract base class for a generic forward solver
 */
class BasePhysics {
public:
  /**
   * @brief Empty constructor
   * @param[in] pmesh An optional mesh reference, must be provided to configure the module
   * @param[in] name Name of the physics module instance
   * when a mesh other than the primary mesh is used
   */
  BasePhysics(std::string name, mfem::ParMesh* pmesh = nullptr);

  /**
   * @brief Constructor that creates n entries in states_ of order p
   *
   * @param[in] n Number of state variables
   * @param[in] p Order of the solver
   * @param[in] pmesh An optional mesh reference, must be provided to configure the module
   * @param[in] name Name of the physics module instance
   * when a mesh other than the default mesh is used
   */
  BasePhysics(int n, int p, std::string name, mfem::ParMesh* pmesh = nullptr);

  /**
   * @brief Construct a new Base Physics object (copy constructor)
   *
   * @param other The other base physics to copy from
   */
  BasePhysics(BasePhysics&& other) = default;

  /**
   * @brief Get the current forward-solution time
   *
   * @return The current forward-solution time
   */
  virtual double time() const;

  /**
   * @brief Get the current forward-solution cycle iteration number
   *
   * @return The current forward-solution cycle iteration number
   */
  virtual int cycle() const;

  /**
   * @brief Set the timestep size (delta time) for the underlying differential equation evolution algorithm
   *
   * @param dt The timestep size to use when advancing the physics module
   */
  virtual void setTimestep(double dt);

  /**
   * @brief Complete the setup and allocate the necessary data structures
   *
   * This finializes the underlying data structures in a solver and
   * enables it to be run through a timestepping loop.
   */
  virtual void completeSetup() = 0;

  /**
   * @brief Accessor for getting named finite element state fields from the physics modules
   *
   * @param state_name The name of the Finite Element State to retrieve
   * @return The named Finite Element State
   */
  virtual const FiniteElementState& state(const std::string& state_name) = 0;

  /**
   * @brief Get a vector of the finite element state solution variable names
   *
   * @return The solution variable names
   */
  virtual std::vector<std::string> stateNames() = 0;

  /**
   * @brief Generate a finite element state object for the given parameter index
   *
   * @param parameter_index The index of the parameter to generate
   * @param parameter_name The name of the parameter to generate
   *
   * @note The user is responsible for managing the lifetime of this object. It is required
   * to exist whenever advanceTimestep, solveAdjoint, or computeSensitivity is called.
   *
   * @note The finite element space for this object is generated from the parameter
   * discretization space (e.g. L2, H1) and the computational mesh given in the physics module constructor.
   *
   * @note The memory address of this parameter is stored in the physics module. If the FiniteElementState
   * pointed to by the returned pointer is modified, the updated parameter value will be used in the physics module.
   */
  std::unique_ptr<FiniteElementState> generateParameter(const std::string& parameter_name, size_t parameter_index);

  /**
   * @brief Register an externally-constructed FiniteElementState object as the source of values for parameter `i`
   *
   * @param parameter_state the values to use for the specified parameter
   * @param parameter_index the index of the parameter
   *
   * @pre The discretization space and mesh for this finite element state must be consistent with the arguments
   * provided in the physics module constructor.
   *
   * @note The memory address of this parameter is stored in the physics module. If the FiniteElementState
   * given in the argument is modified, the updated parameter value will be used in the physics module.
   */
  void registerParameter(const size_t parameter_index, FiniteElementState& parameter_state);

  /**
   * @brief Get a mutable reference to the parameter field of the physics module
   *
   * @param parameter_index The parameter index to retrieve
   * @return The FiniteElementState representing the user-defined parameter
   *
   * @note If the FiniteElementState returned by this function is modified, the updated parameter value will be used in
   * the physics module.
   */
  FiniteElementState& parameter(const size_t parameter_index)
  {
    SLIC_ERROR_ROOT_IF(
        parameter_index >= parameters_.size(),
        axom::fmt::format("Parameter index '{}' is not available in physics module '{}'", parameter_index, name_));

    SLIC_ERROR_ROOT_IF(
        !parameters_[parameter_index].state,
        axom::fmt::format("Parameter index '{}' is not set in physics module '{}'", parameter_index, name_));
    return *parameters_[parameter_index].state;
  }

  /// @overload
  const FiniteElementState& parameter(size_t parameter_index) const
  {
    SLIC_ERROR_ROOT_IF(
        parameter_index >= parameters_.size(),
        axom::fmt::format("Parameter index '{}' is not available in physics module '{}'", parameter_index, name_));

    SLIC_ERROR_ROOT_IF(
        !parameters_[parameter_index].state,
        axom::fmt::format("Parameter index '{}' is not set in physics module '{}'", parameter_index, name_));
    return *parameters_[parameter_index].state;
  }

  /**
   * @brief Get a mutable reference the shape displacement of the associated mesh for this physics object
   *
   * @return The associated shape displacement
   *
   * @note If the FiniteElementState returned by this function is modified, the updated shape displacement will be used
   * in the physics module.
   */
  FiniteElementState& shapeDisplacement() { return shape_displacement_; }

  /// @overload
  const FiniteElementState& shapeDisplacement() const { return shape_displacement_; }

  /**
   * @brief Compute the implicit sensitivity of the quantity of interest used in defining the adjoint load with respect
   * to the parameter field (d QOI/d state * d state/d parameter).
   *
   * @return The sensitivity of the QOI (given implicitly by the adjoint load) with respect to the parameter
   *
   * @pre completeSetup(), advanceTimestep(), and solveAdjoint() must be called prior to this method.
   */
  virtual FiniteElementDual& computeSensitivity(size_t /* parameter_index */)
  {
    SLIC_ERROR_ROOT(axom::fmt::format("Parameter sensitivities not enabled in physics module {}", name_));
    return *parameters_[0].sensitivity;
  }

  /**
   * @brief Compute the implicit sensitivity of the quantity of interest used in defining the adjoint load with respect
   * to the shape displacement field (d QOI/d state * d state/d shape displacement).
   *
   * @return The sensitivity with respect to the shape displacement
   *
   * @pre completeSetup(), advanceTimestep(), and solveAdjoint() must be called prior to this method.
   */
  virtual FiniteElementDual& computeShapeSensitivity()
  {
    SLIC_ERROR_ROOT(axom::fmt::format("Shape sensitivities not enabled in physics module {}", name_));
    return shape_displacement_sensitivity_;
  }

  /**
   * @brief Advance the state variables according to the chosen time integrator and timestep
   *
   * @pre setTimestep() and completeSetup() must be called prior to this method.
   */
  virtual void advanceTimestep() = 0;

  /**
   * @brief Solve the adjoint problem
   * @pre It is expected that the forward analysis is complete and the current states are valid
   * @note If the essential boundary state for the adjoint is not specified, homogeneous essential boundary conditions
   * are applied
   *
   * @return The computed adjoint finite element states
   */
  virtual const std::unordered_map<std::string, const serac::FiniteElementState&> solveAdjoint(
      std::unordered_map<std::string, const serac::FiniteElementDual&> /* adjoint_loads */,
      std::unordered_map<std::string, const serac::FiniteElementState&> /* adjoint_with_essential_boundary */ = {})
  {
    SLIC_ERROR_ROOT(axom::fmt::format("Adjoint analysis not defined for physics module {}", name_));

    // Return a dummy state value to quiet the compiler. This will never get used.
    return {};
  }

  /**
   * @brief Output the current state of the PDE fields in Sidre format and optionally in Paraview format
   *  if \p paraview_output_dir is given.
   *
   * @param[in] paraview_output_dir Optional output directory for paraview visualization files
   */
  virtual void outputStateToDisk(std::optional<std::string> paraview_output_dir = {}) const;

  /**
   * @brief Initializes the Sidre structure for simulation summary data
   *
   * @param[in] datastore Sidre DataStore where data are saved
   * @param[in] t_final Final time of the simulation
   * @param[in] dt The time step
   */
  virtual void initializeSummary(axom::sidre::DataStore& datastore, const double t_final, const double dt) const;

  /**
   * @brief Saves the summary data to the Sidre Datastore
   *
   * @param[in] datastore Sidre DataStore where curves are saved
   * @param[in] t The current time of the simulation
   */
  virtual void saveSummary(axom::sidre::DataStore& datastore, const double t) const;

  /**
   * @brief Destroy the Base Solver object
   */
  virtual ~BasePhysics() = default;

  /**
   * @brief Returns a reference to the mesh object
   */
  const mfem::ParMesh& mesh() const { return mesh_; }

protected:
  /// @brief Name of the physics module
  std::string name_ = {};

  /// @brief ID of the corresponding MFEMSidreDataCollection (denoting a mesh)
  std::string sidre_datacoll_id_ = {};

  /**
   * @brief The primary mesh
   */
  mfem::ParMesh& mesh_;

  /**
   * @brief The MPI communicator
   */
  MPI_Comm comm_;

  /**
   * @brief List of finite element primal states associated with this physics module
   */
  std::vector<const serac::FiniteElementState*> states_;

  /**
   * @brief List of finite element duals associated with this physics module
   */
  std::vector<const serac::FiniteElementDual*> duals_;

  /// @brief The information needed for the physics parameters stored as Finite Element State fields
  struct ParameterInfo {
    /// The trial spaces used for the Functional object
    std::unique_ptr<mfem::ParFiniteElementSpace> trial_space;

    /// The collections needed for the parameter finite element space
    std::unique_ptr<mfem::FiniteElementCollection> trial_collection;

    /// The finite element states representing user-defined and owned parameter fields
    serac::FiniteElementState* state;

    /// The finite element state representing the parameter at the previous evaluation
    std::unique_ptr<serac::FiniteElementState> previous_state;

    /**
     * @brief The sensitivities (dual vectors) of the QOI encoded in the adjoint load with respect to each of the input
     * parameter fields
     * @note this is optional as FiniteElementDuals are not default constructable and
     * we want to set this during the registerParameter or generateParameter method.
     * @note This quantity is also called the vector-Jacobian product during back propagation in data science.
     */
    std::optional<serac::FiniteElementDual> sensitivity;
  };

  /// @brief A vector of the parameters associated with this physics module
  std::vector<ParameterInfo> parameters_;

  /// @brief The parameter info associated with the shape displacement field
  /// @note This is owned by the State Manager since it is associated with the mesh
  FiniteElementState& shape_displacement_;

  /// @brief Sensitivity with respect to the shape displacement field
  /// @note This is owned by the State Manager since it is associated with the mesh
  /// @note This quantity is also called the vector-Jacobian product during back propagation in data science.
  FiniteElementDual& shape_displacement_sensitivity_;

  /**
   *@brief Whether the simulation is time-independent
   */
  bool is_quasistatic_ = true;

  /**
   * @brief Number of significant figures to output for floating-point
   */
  static constexpr int FLOAT_PRECISION_ = 8;

  /**
   * @brief Current time for the forward pass
   */
  double time_;

  /**
   * @brief Current cycle (forward pass time iteration count)
   */
  int cycle_;

  /**
   * @brief Current timestep for the forward pass
   */
  double timestep_;

  /**
   * @brief The value of time at which the ODE solver wants to evaluate the residual
   */
  double ode_time_point_;

  /**
   * @brief MPI rank
   */
  int mpi_rank_;

  /**
   * @brief MPI size
   */
  int mpi_size_;

  /**
   * @brief Order of basis functions
   */
  int order_;

  /**
   * @brief DataCollection pointer for optional paraview output
   */
  mutable std::unique_ptr<mfem::ParaViewDataCollection> paraview_dc_;

  /**
   * @brief State variable initialization indicator
   */
  std::vector<bool> gf_initialized_;

  /**
   * @brief Boundary condition manager instance
   */
  BoundaryConditionManager bcs_;
};

namespace detail {
/**
 * @brief Prepends a prefix to a target string if @p name is non-empty with an
 * underscore delimiter
 * @param[in] prefix The string to prepend
 * @param[in] target The string to prepend to
 */
std::string addPrefix(const std::string& prefix, const std::string& target);
}  // namespace detail

}  // namespace serac

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
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/common.hpp"

namespace serac {

namespace detail {
/**
 * @brief Prepends a prefix to a target string if @p name is non-empty with an
 * underscore delimiter
 * @param[in] prefix The string to prepend
 * @param[in] target The string to prepend to
 */
std::string addPrefix(const std::string& prefix, const std::string& target);
}  // namespace detail

/**
 * @brief This is the abstract base class for a generic forward solver
 */
class BasePhysics {
public:
  /**
   * @brief Empty constructor
   * @param[in] pmesh An optional mesh reference, must be provided to configure the module
   * @param[in] physics_name Name of the physics module instance
   * when a mesh other than the primary mesh is used
   */
  BasePhysics(std::string physics_name, std::string mesh_tag);

  /**
   * @brief Constructor that creates n entries in states_ of order p
   *
   * @param[in] n Number of state variables
   * @param[in] p Order of the solver
   * @param[in] pmesh An optional mesh reference, must be provided to configure the module
   * @param[in] phsyics_name Name of the physics module instance
   * when a mesh other than the default mesh is used
   */
  BasePhysics(int n, int p, std::string physics_name, std::string mesh_tag);

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
  void setParameter(const size_t parameter_index, const FiniteElementState& parameter_state);

  void setShapeDisplacement(const FiniteElementState& shape_displacement);

  /**
   * @brief Compute the implicit sensitivity of the quantity of interest used in defining the adjoint load with respect
   * to the parameter field (d QOI/d state * d state/d parameter).
   *
   * @return The sensitivity of the QOI (given implicitly by the adjoint load) with respect to the parameter
   *
   * @pre completeSetup(), advanceTimestep(), and solveAdjoint() must be called prior to this method.
   */
  virtual const FiniteElementDual& computeSensitivity(size_t /* parameter_index */)
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
  virtual const FiniteElementDual& computeShapeSensitivity()
  {
    SLIC_ERROR_ROOT(axom::fmt::format("Shape sensitivities not enabled in physics module {}", name_));
    return *shape_displacement_sensitivity_;
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
  std::string mesh_tag_ = {};

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
    template <typename FunctionSpace>
    ParameterInfo(mfem::ParMesh& mesh, FunctionSpace space, const std::string& name = "")
    {
      state          = std::make_unique<FiniteElementState>(mesh, space, name);
      previous_state = std::make_unique<FiniteElementState>(mesh, space, "previous_" + name);
      sensitivity    = std::make_unique<FiniteElementDual>(mesh, space, name + "_sensitivity");
      StateManager::storeState(*state);
      StateManager::storeDual(*sensitivity);
    }

    /// The finite element states representing user-defined and owned parameter fields
    std::unique_ptr<serac::FiniteElementState> state;

    /// The finite element state representing the parameter at the previous evaluation
    std::unique_ptr<serac::FiniteElementState> previous_state;

    /**
     * @brief The sensitivities (dual vectors) of the QOI encoded in the adjoint load with respect to each of the input
     * parameter fields
     * @note This quantity is also called the vector-Jacobian product during back propagation in data science.
     */
    std::unique_ptr<serac::FiniteElementDual> sensitivity;
  };

  /// @brief A vector of the parameters associated with this physics module
  std::vector<ParameterInfo> parameters_;

  /// @brief The parameter info associated with the shape displacement field
  /// @note This is owned by the State Manager since it is associated with the mesh
  FiniteElementState& shape_displacement_;

  /// @brief Sensitivity with respect to the shape displacement field
  /// @note This quantity is also called the vector-Jacobian product during back propagation in data science.
  /// @note This is owned by the physics instance as the sensitivity is with respect to a certain PDE residual (i.e.
  /// physics module)
  std::unique_ptr<FiniteElementDual> shape_displacement_sensitivity_;

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

}  // namespace serac

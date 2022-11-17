// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
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
   * @brief Set the current time
   *
   * @param[in] time The time
   */
  virtual void setTime(const double time);

  /**
   * @brief Get the current time
   *
   * @return The current time
   */
  virtual double time() const;

  /**
   * @brief Set the current cycle
   *
   * @param[in] cycle The cycle
   */
  virtual void setCycle(const int cycle);

  /**
   * @brief Get the current cycle
   *
   * @return The current cycle
   */
  virtual int cycle() const;

  /**
   * @brief Complete the setup and allocate the necessary data structures
   *
   * This finializes the underlying MFEM data structures in a solver and
   * enables it to be run through a timestepping loop
   */
  virtual void completeSetup() = 0;

  /**
   * @brief Accessor for getting named finite element state fields from the physics modules
   *
   * @param state_name The name of the Finite Element State to retrieve
   * @return The named Finite Element State
   */
  virtual const FiniteElementState& getState(const std::string& state_name) = 0;

  /**
   * @brief Get a vector of the finite element state solution variable names
   *
   * @return The solution variable names
   */
  virtual std::vector<std::string> getStateNames() = 0;

  /**
   * @brief Generate a finite element state object for the given parameter index
   *
   * @param parameter_index The index of the parameter to generate
   * @param parameter_name The name of the parameter to generate
   *
   * @note The user is responsible for managing the lifetime of this object. It is required
   * to exist whenever this physics module is called.
   */
  virtual std::unique_ptr<FiniteElementState> generateParameter(size_t             parameter_index,
                                                                const std::string& parameter_name);

  /**
   * @brief register the provided FiniteElementState object as the source of values for parameter `i`
   *
   * @param parameter_state the values to use for the specified parameter
   * @param parameter_index the index of the parameter
   */
  void setParameter(FiniteElementState& parameter_state, size_t parameter_index);

  /**
   * @brief Get the parameter field of the physics module
   *
   * @param parameter_index The parameter index to retrieve
   * @return The FiniteElementState representing the user-defined parameter
   */
  virtual FiniteElementState& getParameter(size_t parameter_index)
  {
    SLIC_ERROR_ROOT_IF(
        parameter_index >= parameter_info_.size(),
        axom::fmt::format("Parameter index {} is not available in physics module {}", parameter_index, name_));
    SLIC_ERROR_ROOT_IF(!parameter_info_[parameter_index].state,
                       axom::fmt::format("Parameter index {} is not set in physics module {}", parameter_index, name_));
    return *parameter_info_[parameter_index].state;
  }

  /// @overload
  virtual const FiniteElementState& getParameter(size_t parameter_index) const
  {
    SLIC_ERROR_ROOT_IF(
        parameter_index >= parameter_info_.size(),
        axom::fmt::format("Parameter index {} is not available in physics module {}", parameter_index, name_));
    SLIC_ERROR_ROOT_IF(!parameter_info_[parameter_index].state,
                       axom::fmt::format("Parameter index {} is not set in physics module {}", parameter_index, name_));
    return *parameter_info_[parameter_index].state;
  }

  /**
   * @brief Advance the state variables according to the chosen time integrator
   *
   * @param[inout] dt The timestep to advance. For adaptive time integration methods, the actual timestep is returned.
   */
  virtual void advanceTimestep(double& dt) = 0;

  /**
   * @brief Solve the adjoint problem
   * @pre It is expected that the forward analysis is complete and the current state is valid
   * @note If the essential boundary state is not specified, homogeneous essential boundary conditions are applied
   *
   * @return The computed adjoint finite element state
   */
  virtual const serac::FiniteElementState& solveAdjoint(FiniteElementDual& /*adjoint_load */,
                                                        FiniteElementDual* /* dual_with_essential_boundary */ = nullptr)
  {
    SLIC_ERROR_ROOT(axom::fmt::format("Adjoint analysis not defined for physics module {}", name_));

    // Return a dummy state value to quiet the compiler. This will never get used.
    return *states_[0];
  }

  /**
   * @brief Output the current state of the PDE fields in Sidre format and optionally in Paraview format
   *  if \p paraview_output_dir is given.
   *
   * @param[in] paraview_output_dir Optional output directory for paraview visualization files
   */
  virtual void outputState(std::optional<std::string> paraview_output_dir = {}) const;

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
   * @brief List of finite element states associated with this physics module
   */
  std::vector<serac::FiniteElementState*> states_;

  /**
   * @brief List of finite element duals associated with this physics module
   */
  std::vector<serac::FiniteElementDual*> duals_;

  /// @brief The information needed for the physics parameters stored as Finite Element State fields
  struct ParameterInfo {
    /// The trial spaces used for the Functional object
    std::unique_ptr<mfem::ParFiniteElementSpace> trial_space;

    /// The finite element states representing user-defined and owned parameter fields
    serac::FiniteElementState* state;

    /**
     * @brief The sensitivities (dual vectors) with repect to each of the input parameter fields
     * @note this optional as FiniteElementDuals are not default constructable and
     * we want to set this during the setParameter or generateParameter method.
     */
    std::optional<FiniteElementDual> sensitivity;
  };

  /// @brief A vector of the parameters associated with this physics module
  std::vector<ParameterInfo> parameter_info_;

  /**
   *@brief Whether the simulation is time-independent
   */
  bool is_quasistatic_ = true;

  /**
   * @brief Number of significant figures to output for floating-point
   */
  static constexpr int FLOAT_PRECISION_ = 8;

  /**
   * @brief Current time
   */
  double time_;

  /**
   * @brief Current cycle
   */
  int cycle_;

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

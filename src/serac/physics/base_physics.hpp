// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
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

#include "serac/physics/utilities/boundary_condition_manager.hpp"
#include "serac/physics/utilities/equation_solver.hpp"
#include "serac/physics/utilities/finite_element_state.hpp"

namespace serac {

/**
 * @brief This is the abstract base class for a generic forward solver
 */
class BasePhysics {
public:
  /**
   * @brief Empty constructor
   */
  BasePhysics();

  /**
   * @brief Constructor that creates n entries in state_ of order p
   *
   * @param[in] n Number of state variables
   * @param[in] p Order of the solver
   */
  BasePhysics(int n, int p);

  /**
   * @brief Construct a new Base Physics object (copy constructor)
   *
   * @param other The other base physics to copy from
   */
  BasePhysics(BasePhysics&& other) = default;

  /**
   * @brief Set a list of true degrees of freedom from a coefficient
   *
   * @param[in] true_dofs The true degrees of freedom to set with a Dirichlet condition
   * @param[in] ess_bdr_coef The coefficient that evaluates to the Dirichlet condition
   * @param[in] component The component to set (-1 implies all components are set)
   */
  virtual void setTrueDofs(const mfem::Array<int>& true_dofs, serac::GeneralCoefficient ess_bdr_coef,
                           const int component = -1);

  /**
   * @brief Get the list of state variable grid functions
   *
   * @return the current vector of finite element states
   */
  virtual const std::vector<std::reference_wrapper<serac::FiniteElementState>>& getState() const;

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
   * @brief Advance the state variables according to the chosen time integrator
   *
   * @param[inout] dt The timestep to advance. For adaptive time integration methods, the actual timestep is returned.
   */
  virtual void advanceTimestep(double& dt) = 0;

  /**
   * @brief Initialize the state variable output
   *
   * @param[in] output_type The type of output files to produce
   * @param[in] root_name The root name of the output files
   */
  virtual void initializeOutput(const serac::OutputType output_type, const std::string& root_name);

  /**
   * @brief Output the current state of the PDE fields
   *
   */
  virtual void outputState() const;

  /**
   * @brief Destroy the Base Solver object
   */
  virtual ~BasePhysics() = default;

  /**
   * @brief Returns a reference to the mesh object
   */
  const mfem::ParMesh& mesh() const { return mesh_; }

protected:
  /**
   * @brief The primary mesh
   */
  mfem::ParMesh& mesh_;

  /**
   * @brief The MPI communicator
   */
  MPI_Comm comm_;

  /**
   * @brief List of finite element data structures
   */
  std::vector<std::reference_wrapper<serac::FiniteElementState>> state_;

  /**
   * @brief Block vector storage of the true state
   */
  std::unique_ptr<mfem::BlockVector> block_;

  /**
   * @brief Type of state variable output
   */
  serac::OutputType output_type_;

  /**
   *@brief Whether the simulation is time-independent
   */
  bool is_quasistatic_ = true;

  /**
   * @brief Root output name
   */
  std::string root_name_;

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
   * @brief DataCollection pointer
   */
  std::unique_ptr<mfem::DataCollection> dc_;

  /**
   * @brief State variable initialization indicator
   */
  std::vector<bool> gf_initialized_;

  /**
   * @brief Boundary condition manager instance
   */
  BoundaryConditionManager bcs_;

  /**
   * @brief Flag denoting output initialization
   *
   */
  bool output_is_initialized_ = false;
};

namespace detail {
/**
 * @brief Prepends a prefix to a target string if @p name is non-empty with an
 * underscore delimiter
 * @param[in] addPrefix The string to prepend
 * @param[in] target The string to prepend to
 */
std::string addPrefix(const std::string& prefix, const std::string& target);
}  // namespace detail

}  // namespace serac

// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file base_solver.hpp
 *
 * @brief The base interface class for a generic PDE solver
 */

#ifndef BASE_PHYSICS
#define BASE_PHYSICS

#include <memory>

#include "mfem.hpp"
#include "physics/utilities/boundary_condition_manager.hpp"
#include "physics/utilities/equation_solver.hpp"
#include "physics/utilities/finite_element_state.hpp"

namespace serac {

/**
 * @brief This is the abstract base class for a generic forward solver
 */
class BasePhysics {
public:
  /**
   * @brief Empty constructor
   *
   * @param[in] mesh The primary mesh
   */
  BasePhysics(std::shared_ptr<mfem::ParMesh> mesh);

  /**
   * @brief Constructor that creates n entries in state_ of order p
   *
   * @param[in] mesh The primary mesh
   * @param[in] n Number of state variables
   * @param[in] p Order of the solver
   */
  BasePhysics(std::shared_ptr<mfem::ParMesh> mesh, int n, int p);

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
   * @brief Set the state variables from a vector of coefficients
   *
   * @param[in] state_coef A vector of coefficients to project on the state grid functions
   */
  virtual void setState(const std::vector<serac::GeneralCoefficient>& state_coef);

  /**
   * @brief Set the state variables from an existing grid function
   *
   * @param[in] state A vector of finite element states to initialze the solver
   */
  virtual void setState(const std::vector<std::shared_ptr<serac::FiniteElementState> >& state);

  /**
   * @brief Get the list of state variable grid functions
   *
   * @return the current vector of finite element states
   */
  virtual std::vector<std::shared_ptr<serac::FiniteElementState> > getState() const;

  /**
   * @brief Set the time integration method
   *
   * @param[in] timestepper The timestepping method for the solver
   */
  virtual void setTimestepper(const serac::TimestepMethod timestepper);

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

protected:
  /**
   * @brief The MPI communicator
   */
  MPI_Comm comm_;

  /**
   * @brief The primary mesh
   */
  std::shared_ptr<mfem::ParMesh> mesh_;

  /**
   * @brief List of finite element data structures
   */
  std::vector<std::shared_ptr<serac::FiniteElementState> > state_;

  /**
   * @brief Block vector storage of the true state
   */
  std::unique_ptr<mfem::BlockVector> block_;

  /**
   * @brief Type of state variable output
   */
  serac::OutputType output_type_;

  /**
   *@brief Time integration method
   */
  serac::TimestepMethod timestepper_ = TimestepMethod::QuasiStatic;

  /**
   * @brief MFEM ode solver object
   */
  std::unique_ptr<mfem::ODESolver> ode_solver_;

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
};

}  // namespace serac

#endif

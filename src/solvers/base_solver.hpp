// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef BASE_SOLVER
#define BASE_SOLVER

#include <map>
#include <memory>

#include "common/serac_types.hpp"
#include "mfem.hpp"

/// This is the abstract base class for a generic forward solver
class BaseSolver {
 private:
  void RegisterEssentialBC(std::shared_ptr<BoundaryCondition> bc, const std::set<int> &ess_bdr, 
                           mfem::ParFiniteElementSpace &fes, int component);

  void RegisterNaturalBC(std::shared_ptr<BoundaryCondition> bc, const std::set<int> &nat_bdr, 
                         int component);

 protected:
  /// The MPI communicator
  MPI_Comm m_comm;

  /// List of finite element data structures
  std::vector<std::shared_ptr<FiniteElementState> > m_state;

  /// Block vector storage of the true state
  std::unique_ptr<mfem::BlockVector> m_block;

  /// Essential BC markers
  std::vector<std::shared_ptr<BoundaryCondition> > m_ess_bdr;

  /// Natural BC markers
  std::vector<std::shared_ptr<BoundaryCondition> > m_nat_bdr;

  /// Type of state variable output
  OutputType m_output_type;

  /// Time integration method
  TimestepMethod m_timestepper;

  /// MFEM ode solver object
  std::unique_ptr<mfem::ODESolver> m_ode_solver;

  /// Root output name
  std::string m_root_name;

  /// Current time
  double m_time;

  /// Current cycle
  int m_cycle;

  /// MPI rank
  int m_rank;

  /// VisIt data collection pointer
  std::unique_ptr<mfem::VisItDataCollection> m_visit_dc;

  /// State variable initialization indicator
  std::vector<bool> m_gf_initialized;

 public:
  /// Empty constructor
  BaseSolver(MPI_Comm comm);

  /// Constructor that creates n entries in m_state
  BaseSolver(MPI_Comm comm, int n);

  /// Set the essential boundary conditions from a list of boundary markers and
  /// a coefficient
  virtual void SetEssentialBCs(const std::set<int> &ess_bdr, std::shared_ptr<mfem::Coefficient> ess_bdr_coef,
                               mfem::ParFiniteElementSpace &fes, int component = -1);

  /// Set the vector-valued essential boundary conditions from a list of
  /// boundary markers and a coefficient
  virtual void SetEssentialBCs(const std::set<int> &                 ess_bdr,
                               std::shared_ptr<mfem::VectorCoefficient> ess_bdr_vec_coef,
                               mfem::ParFiniteElementSpace &fes, int component = -1);

  /// Set a list of true degrees of freedom from a coefficient
  virtual void SetTrueDofs(const mfem::Array<int> &true_dofs, std::shared_ptr<mfem::Coefficient> ess_bdr_coef);

  /// Set a list of true degrees of freedom from a vector coefficient
  virtual void SetTrueDofs(const mfem::Array<int> &                 true_dofs,
                           std::shared_ptr<mfem::VectorCoefficient> ess_bdr_vec_coef);

  /// Set the natural boundary conditions from a list of boundary markers and a
  /// coefficient
  virtual void SetNaturalBCs(const std::set<int> &nat_bdr, std::shared_ptr<mfem::Coefficient> nat_bdr_coef,
                             int component = -1);

  /// Set the vector-valued natural boundary conditions from a list of boundary
  /// markers and a coefficient
  virtual void SetNaturalBCs(const std::set<int> &nat_bdr, std::shared_ptr<mfem::VectorCoefficient> nat_bdr_vec_coef,
                             int component = -1);

  /// Set the state variables from a coefficient
  virtual void SetState(const std::vector<std::shared_ptr<mfem::Coefficient> > &state_coef);

  /// Set the state variables from a vector coefficient
  virtual void SetState(const std::vector<std::shared_ptr<mfem::VectorCoefficient> > &state_vec_coef);

  /// Set the state variables from an existing grid function
  virtual void SetState(const std::vector<std::shared_ptr<FiniteElementState> > &state);

  /// Get the list of state variable grid functions
  virtual std::vector<std::shared_ptr<FiniteElementState> > GetState() const;

  /// Set the time integration method
  virtual void SetTimestepper(TimestepMethod timestepper);

  /// Set the current time
  virtual void SetTime(const double time);

  /// Get the current time
  virtual double GetTime() const;

  /// Get the current cycle
  virtual int GetCycle() const;

  /// Complete the setup and allocate the necessary data structures
  virtual void CompleteSetup() = 0;

  /// Advance the state variables according to the chosen time integrator
  virtual void AdvanceTimestep(double &dt) = 0;

  /// Initialize the state variable output
  virtual void InitializeOutput(const OutputType output_type, const std::string &root_name);

  /// output the state variables
  virtual void OutputState() const;

  /// Destructor
  virtual ~BaseSolver() = default;
};

#endif

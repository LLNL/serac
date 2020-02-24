// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef BASE_SOLVER
#define BASE_SOLVER

#include "mfem.hpp"
#include "common/serac_types.hpp"

/// This is the abstract base class for a generic forward solver
class BaseSolver
{
protected:
  /// List of finite element data structures
  mfem::Array<FiniteElementState> m_state;

  /// Essential BC markers
  mfem::Array<int> m_ess_bdr;

  /// Natural BC markers
  mfem::Array<int> m_nat_bdr;

  /// Pointer to the essential BC coefficient
  mfem::Coefficient *m_ess_bdr_coef;

  /// Pointer to the vector-valued essential BC coefficient
  mfem::VectorCoefficient *m_ess_bdr_vec_coef;

  /// Pointer to the nautral BC coefficient
  mfem::Coefficient *m_nat_bdr_coef;

  /// Pointer to the vector-valued natural BC coefficient
  mfem::VectorCoefficient *m_nat_bdr_vec_coef;

  /// Array of the essential degree of freedom indicies
  mfem::Array<int> m_ess_tdof_list;

  /// Type of state variable output
  OutputType m_output_type;

  /// Time integration method
  TimestepMethod m_timestepper;

  /// MFEM ode solver object
  mfem::ODESolver *m_ode_solver;

  /// Root output name
  std::string m_root_name;

  /// Current time
  double m_time;

  /// Current cycle
  int m_cycle;

  /// MPI rank
  int m_rank;

  /// VisIt data collection pointer
  mfem::VisItDataCollection* m_visit_dc;

  /// State variable initialization indicator
  bool m_gf_initialized;

public:
  /// Empty constructor
  BaseSolver();

  /// Constructor from previously constructed grid function
  BaseSolver(mfem::Array<FiniteElementState> &state);

  /// Set the essential boundary conditions from a list of boundary markers and a coefficient
  virtual void SetEssentialBCs(mfem::Array<int> &ess_bdr, mfem::Coefficient *ess_bdr_coef);

  /// Set the vector-valued essential boundary conditions from a list of boundary markers and a coefficient
  virtual void SetEssentialBCs(mfem::Array<int> &ess_bdr, mfem::VectorCoefficient *ess_bdr_vec_coef);

  /// Set the natural boundary conditions from a list of boundary markers and a coefficient
  virtual void SetNaturalBCs(mfem::Array<int> &nat_bdr, mfem::Coefficient *nat_bdr_coef);

  /// Set the vector-valued natural boundary conditions from a list of boundary markers and a coefficient
  virtual void SetNaturalBCs(mfem::Array<int> &nat_bdr, mfem::VectorCoefficient *nat_bdr_vec_coef);

  /// Set the state variables from an existing grid function
  virtual void SetState(const mfem::Array<FiniteElementState> &state);

  /// Get the list of state variable grid functions
  virtual mfem::Array<FiniteElementState> GetState() const;

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
  virtual void InitializeOutput(const OutputType output_type, const std::string root_name,
                                const mfem::Array<std::string> names);

  /// output the state variables
  virtual void OutputState() const;

  /// Destructor
  virtual ~BaseSolver();

};


#endif

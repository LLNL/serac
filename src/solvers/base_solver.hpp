// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
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
  /// List of finite element collections for the state variables
  mfem::Array<const mfem::FiniteElementCollection*> m_fecolls;

  /// List of finite element spaces for the state variables
  mfem::Array<mfem::ParFiniteElementSpace*> m_fespaces;

  /// List of state variable grid functions
  mfem::Array<mfem::ParGridFunction*> m_state_gf;

  /// List of true degree of freedom vectors for the state variables
  mfem::Array<mfem::HypreParVector*> m_true_vec;

  /// Pointer to the parallel mesh object
  mfem::ParMesh *m_pmesh;

  /// Essential BC markers
  mfem::Array<int> m_ess_bdr;

  /// Natural BC markers
  mfem::Array<int> m_nat_bdr;

  /// Pointer to the essential BC coefficient
  const mfem::Coefficient *m_ess_bdr_coef;

  /// Pointer to the nautral BC coefficient
  const mfem::Coefficient *m_nat_bdr_coef;

  /// Array of the essential degree of freedom indicies
  mfem::Array<int> m_ess_tdof_list;

  /// Type of state variable output
  OutputType m_output_type;

  /// Time integration method
  TimestepMethod m_timestepper;

  /// MFEM ode solver object
  mfem::ODESolver *m_ode_solver;

  /// Array of variable names
  mfem::Array<std::string> m_state_names;

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
  BaseSolver(mfem::Array<mfem::ParGridFunction*> &stategf);

  /// Set the essential boundary conditions from a list of boundary markers and a coefficient
  virtual void SetEssentialBCs(mfem::Array<int> &ess_bdr, mfem::Coefficient *ess_bdr_coef);

  /// Set the natural boundary conditions from a list of boundary markers and a coefficient
  virtual void SetNaturalBCs(mfem::Array<int> &nat_bdr, mfem::Coefficient *nat_bdr_coef);

  /// Set the state variables from an existing grid function
  virtual void SetState(const mfem::Array<mfem::ParGridFunction*> &state_gf);

  /// Get the list of state variable grid functions
  virtual mfem::Array<mfem::ParGridFunction*> GetState() const;

  /// Set the time integration method
  virtual void SetTimestepper(TimestepMethod timestepper);

  /// Set the current time
  virtual void SetTime(const double time);

  /// Get the current time
  virtual double GetTime() const;

  /// Get the current cycle
  virtual int GetCycle() const;

  /// Advance the state variables according to the chosen time integrator
  virtual void AdvanceTimestep(double &dt) = 0;

  /// Initialize the state variable output
  virtual void InitializeOutput(const OutputType output_type, const mfem::Array<std::string> names);

  /// Output the state variables
  virtual void OutputState() const;

  /// Destructor
  virtual ~BaseSolver();

};


#endif

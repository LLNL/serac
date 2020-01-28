// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef BASE_SOLVER
#define BASE_SOLVER

#include "mfem.hpp"
#include "common/serac_types.hpp"

// This is the abstract base class for a generic forward solver

class BaseSolver
{
protected:
  mfem::Array<const mfem::FiniteElementCollection*> m_fecolls;
  mfem::Array<mfem::ParFiniteElementSpace*> m_fespaces;
  mfem::Array<mfem::ParGridFunction*> m_state_gf;
  mfem::Array<mfem::HypreParVector*> m_true_vec;
  mfem::ParMesh *m_pmesh;
  mfem::Array<int> m_ess_bdr;
  mfem::Array<int> m_nat_bdr;
  const mfem::Coefficient *m_ess_bdr_coef;
  const mfem::Coefficient *m_nat_bdr_coef;
  mfem::Array<int> m_ess_tdof_list;

  OutputType m_output_type;

  TimestepMethod m_timestepper;
  mfem::ODESolver *m_ode_solver;

  mfem::Array<std::string> m_state_names;

  double m_time;
  int m_cycle;
  int m_rank;

  mfem::VisItDataCollection* m_visit_dc;

public:
  explicit BaseSolver();

  explicit BaseSolver(mfem::Array<mfem::ParGridFunction*> &stategf);

  virtual void SetEssentialBCs(mfem::Array<int> &ess_bdr, mfem::Coefficient *ess_bdr_coef);

  virtual void SetNaturalBCs(mfem::Array<int> &nat_bdr, mfem::Coefficient *nat_bdr_coef);

  virtual void SetState(const mfem::Array<mfem::ParGridFunction*> &state_gf);

  virtual mfem::Array<mfem::ParGridFunction*> GetState() const;

  virtual void SetTimestepper(TimestepMethod timestepper);

  virtual void SetTime(const double time);

  virtual double GetTime() const;

  virtual int GetCycle() const;

  virtual void StaticSolve() = 0;

  virtual void AdvanceTimestep(double dt) = 0;

  virtual void InitializeOutput(const OutputType output_type, const mfem::Array<std::string> names);

  virtual void OutputState() const;

  virtual ~BaseSolver();

};


#endif
